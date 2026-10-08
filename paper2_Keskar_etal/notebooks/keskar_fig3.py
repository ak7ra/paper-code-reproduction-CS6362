#!/usr/bin/env python3
"""
Reproduction of Figure 3 (panels a: F1, c: C1, d: C2) of
Keskar et al., "On Large-Batch Training for Deep Learning: Generalization Gap
and Sharp Minima" (ICLR 2017).

What this script does (per network):
  1. Trains `--trials` paired small-batch (SB) / large-batch (LB) models with ADAM.
     Each SB/LB pair starts from IDENTICAL initial parameters.
  2. Stops each run with a fixed, test-set-free criterion (see `should_stop`).
  3. Saves every trained SB and LB model (final iterate).
  4. Summarises the 5 trials (cf. paper Table 2) as a sanity check.
  5. Picks ONE pair via a pre-declared seeded random draw (or all pairs with --pairs all).
  6. Evaluates theta(alpha) = alpha * x_LB + (1 - alpha) * x_SB for alpha in [-1, 2]
     on the FULL training and test sets (cross-entropy + accuracy), re-estimating
     BatchNorm running statistics from the training data at every alpha.
  7. Saves raw numbers (.npz / .csv) and the figure (.png / .pdf).

Nothing from the paper's figures is hard-coded. The only paper numbers present in this file
are the Table-2 accuracies in PAPER_TABLE2_REF; they are used ONLY to print an
informational comparison in the sanity checks and never influence training or plots.

Legend for comments:
  [PAPER]   explicitly stated in the paper (page given)
  [INFERRED] inferred from the paper
  [REPRO CHOICE] NOT specified by the paper; a choice made for this reproduction

Examples:
  python keskar_fig3.py --nets F1                       # cheapest
  python keskar_fig3.py --nets F1 C1                    # skip C2
  python keskar_fig3.py --nets C2 --trials 1 --max-epochs 100   # cheaper C2
  python keskar_fig3.py --nets F1 --stage plot          # re-plot from saved raw results
  python keskar_fig3.py --nets F1 C1 C2 --synthetic 2000 --trials 2 --max-epochs 25 \
        --stop-window 5 --n-alpha 31                    # pipeline smoke test, random data
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import platform
import sys
import time

# Needed for deterministic cuBLAS; must be set before CUDA is initialised.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BN_TYPES = nn.modules.batchnorm._BatchNorm

# [PAPER] alpha in [-1, 2] (Sec. 2.2.1, p. 4; Fig. 3 caption p. 6)
ALPHA_MIN, ALPHA_MAX = -1.0, 2.0

# Network -> dataset and the panel of Figure 3 it corresponds to.
NET_CONFIGS = {
    "F1": dict(dataset="mnist", panel="3a"),      # [PAPER] Table 1, App. B.1
    "C1": dict(dataset="cifar10", panel="3c"),    # [PAPER] Table 1, App. B.3
    "C2": dict(dataset="cifar10", panel="3d"),    # [PAPER] Table 1, App. B.4
}

# Paper Table 2 (p. 4), accuracies in %.  INFORMATIONAL SANITY CHECK ONLY.
# Never used for plotting, training, or stopping.
PAPER_TABLE2_REF = {
    #        SB train, SB test, LB train, LB test
    "F1": (99.66, 98.03, 99.92, 97.81),
    "C1": (99.89, 80.04, 99.66, 77.26),
    "C2": (99.99, 89.24, 99.99, 87.26),
}
TABLE2_TOL_PP = 2.0  # [REPRO CHOICE] tolerance (percentage points) for the informational warning


# ----------------------------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------------------------
def log(*a):
    print(*a, flush=True)


def warn(*a):
    print("[WARN]", *a, flush=True)


def set_determinism(seed: int):
    """[REPRO CHOICE] Seed everything; bit-exact reproducibility is only expected on the same
    hardware/software stack (some CUDA kernels, e.g. max-pool backward, are nondeterministic)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # [REPRO CHOICE] Disable TF32 so fp32 results do not depend on Ampere+ defaults.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def to_float(xb: torch.Tensor) -> torch.Tensor:
    # [REPRO CHOICE] The paper says "no pre-processing" (App. A, p. 12) but not whether pixels are
    # rescaled.  We rescale uint8 [0,255] -> float [0,1] ONLY.  No mean/std normalisation,
    # no augmentation ([INFERRED]: augmentation is introduced only as a remedy in App. E.1, p. 13).
    return xb.float().div_(255.0)


# ----------------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------------
def load_data(dataset: str, data_dir: str, device, synthetic: int = 0, seed: int = 0):
    """Returns uint8 tensors on `device`: xtr, ytr, xte, yte.  Images are (N, C, H, W)."""
    if synthetic > 0:
        # Pipeline smoke test ONLY (random images / labels). Not a reproduction of anything.
        warn("Using SYNTHETIC random data: results are meaningless; this only tests the pipeline.")
        g = torch.Generator().manual_seed(seed)
        shape = (1, 28, 28) if dataset == "mnist" else (3, 32, 32)
        n_tr, n_te = synthetic, max(synthetic // 5, 10)
        xtr = torch.randint(0, 256, (n_tr, *shape), generator=g, dtype=torch.uint8)
        xte = torch.randint(0, 256, (n_te, *shape), generator=g, dtype=torch.uint8)
        ytr = torch.randint(0, 10, (n_tr,), generator=g)
        yte = torch.randint(0, 10, (n_te,), generator=g)
    else:
        import torchvision  # imported lazily so --synthetic works without torchvision
        # [PAPER] official train/test splits: MNIST 60000/10000, CIFAR-10 50000/10000 (Table 5, p. 12).
        # [INFERRED] no validation split; the test set is never used for any decision.
        if dataset == "mnist":
            tr = torchvision.datasets.MNIST(data_dir, train=True, download=True)
            te = torchvision.datasets.MNIST(data_dir, train=False, download=True)
            xtr, ytr = tr.data.unsqueeze(1).contiguous(), torch.as_tensor(tr.targets)
            xte, yte = te.data.unsqueeze(1).contiguous(), torch.as_tensor(te.targets)
        elif dataset == "cifar10":
            tr = torchvision.datasets.CIFAR10(data_dir, train=True, download=True)
            te = torchvision.datasets.CIFAR10(data_dir, train=False, download=True)
            xtr = torch.from_numpy(tr.data).permute(0, 3, 1, 2).contiguous()
            xte = torch.from_numpy(te.data).permute(0, 3, 1, 2).contiguous()
            ytr, yte = torch.tensor(tr.targets), torch.tensor(te.targets)
        else:
            raise ValueError(dataset)

        # Sanity checks on the real data.
        exp = {"mnist": (60000, 10000, (1, 28, 28)), "cifar10": (50000, 10000, (3, 32, 32))}[dataset]
        assert xtr.shape[0] == exp[0] and xte.shape[0] == exp[1], "unexpected dataset size"
        assert tuple(xtr.shape[1:]) == exp[2], "unexpected image shape"
    assert xtr.dtype == torch.uint8 and xte.dtype == torch.uint8
    assert int(ytr.min()) >= 0 and int(ytr.max()) <= 9
    ytr, yte = ytr.long(), yte.long()
    return dict(xtr=xtr.to(device), ytr=ytr.to(device), xte=xte.to(device), yte=yte.to(device))


# ----------------------------------------------------------------------------------------
# Architectures
# ----------------------------------------------------------------------------------------
def build_model(net: str) -> nn.Sequential:
    """Architectures from App. B (p. 12).  See per-line tags for what is paper vs. our choice."""
    L = []
    if net == "F1":
        # [PAPER] 784-d input, 5 batch-normalised layers of 512 ReLU units, 10-way softmax output.
        # [REPRO CHOICE] order Dense -> BN -> ReLU; Linear layers keep a bias (framework default);
        #   BN affine=True, framework-default eps/momentum.
        # Softmax is folded into F.cross_entropy (applied to the logits).
        L.append(nn.Flatten())
        d_in = 784
        for _ in range(5):
            L += [nn.Linear(d_in, 512), nn.BatchNorm1d(512), nn.ReLU()]
            d_in = 512
        L.append(nn.Linear(512, 10))
    elif net == "C1":
        # [PAPER] "2 sets of [64,5,5,2]-MaxPool(3)", dense (384, 192), output 10, BN on all layers,
        #   ReLU, dropout with 0.5 retention probability on the two dense layers.
        # [INFERRED] "2 sets" = two conv layers (64 filters, 5x5, stride 2), each followed by a pool.
        # [REPRO CHOICE] conv padding=2 ("same"-style), MaxPool kernel 3 / stride 2 / padding 1,
        #   spatial size 32 -> 16 -> 8 -> 4 -> 2, giving 2*2*64 = 256 features.
        #   Order Conv -> BN -> ReLU -> Pool; Dense -> BN -> ReLU -> Dropout(p_drop=0.5).
        c_in = 3
        for _ in range(2):
            L += [nn.Conv2d(c_in, 64, kernel_size=5, stride=2, padding=2), nn.BatchNorm2d(64),
                  nn.ReLU(), nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
            c_in = 64
        L.append(nn.Flatten())
        d_in = 64 * 2 * 2
        for d_out in (384, 192):
            L += [nn.Linear(d_in, d_out), nn.BatchNorm1d(d_out), nn.ReLU(), nn.Dropout(p=0.5)]
            d_in = d_out
        L.append(nn.Linear(d_in, 10))
    elif net == "C2":
        # [PAPER] 2x[64,3,3,1], 2x[128,3,3,1], 3x[256,3,3,1], 3x[512,3,3,1], 3x[512,3,3,1], MaxPool(2)
        #   after each stack, then a 512-d dense layer and a 10-d output.  The App. B.4 text says
        #   "C3 network" but this is [INFERRED] to be a typo for C2.
        # [PAPER] "activation and properties of each layer as in B.3" -> BN everywhere, ReLU, dropout
        #   (retention 0.5) on the dense layer.
        # [REPRO CHOICE] 3x3 convs with padding=1 ("same"); MaxPool(2) stride 2: 32 -> 1 after five
        #   pools => 512 features; Conv -> BN -> ReLU ordering.
        c_in = 3
        for n_conv, c_out in [(2, 64), (2, 128), (3, 256), (3, 512), (3, 512)]:
            for _ in range(n_conv):
                L += [nn.Conv2d(c_in, c_out, kernel_size=3, stride=1, padding=1),
                      nn.BatchNorm2d(c_out), nn.ReLU()]
                c_in = c_out
            L.append(nn.MaxPool2d(kernel_size=2, stride=2))
        L += [nn.Flatten(), nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(),
              nn.Dropout(p=0.5), nn.Linear(512, 10)]
    else:
        raise ValueError(net)
    return nn.Sequential(*L)


def init_params(model: nn.Module):
    # [PAPER] 5 runs from different "uniformly distributed random" starting points (p. 4).
    # [REPRO CHOICE] Glorot-uniform weights, zero biases, BN gamma=1 / beta=0.
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, BN_TYPES):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


def n_params(model):
    return sum(p.numel() for p in model.parameters())


# ----------------------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor, bs: int):
    """[INFERRED] Full-dataset mean cross-entropy (Eq. 1, p. 1) and accuracy (%), eval mode
    (dropout off, BN uses stored running statistics).  Loss computed from logits with a stable
    log-softmax, no clipping (Fig. 3 shows loss values far above 1)."""
    model.eval()
    n = x.shape[0]
    tot_loss, correct = 0.0, 0
    for i in range(0, n, bs):
        out = model(to_float(x[i:i + bs])).float()
        yb = y[i:i + bs]
        tot_loss += F.cross_entropy(out, yb, reduction="sum").item()
        correct += (out.argmax(1) == yb).sum().item()
    return tot_loss / n, 100.0 * correct / n


@torch.no_grad()
def reestimate_bn(model: nn.Module, x_train: torch.Tensor, bs: int):
    """[REPRO CHOICE, option (B) of my reconstruction] Re-estimate BN running statistics with ONE
    pass over the full TRAINING set, using exact aggregated per-channel mean and (biased) variance
    (sums / sums of squares accumulated in float64), i.e. no exponential moving average.

    During this pass BN layers normalise with batch statistics (train mode, as during training)
    while dropout is off; the statistics themselves are pooled over the whole training set.
    Training data is used even when the test loss is later evaluated (no test leakage)."""
    bns = [m for m in model.modules() if isinstance(m, BN_TYPES)]
    if not bns:
        return
    acc = {m: [torch.zeros(m.num_features, dtype=torch.float64, device=x_train.device),
               torch.zeros(m.num_features, dtype=torch.float64, device=x_train.device), 0]
           for m in bns}

    def make_hook(m):
        def hook(mod, inp):
            t = inp[0]
            dims = [0] if t.dim() == 2 else [0, 2, 3]
            s = acc[m]
            s[0] += t.sum(dims, dtype=torch.float64)
            s[1] += (t * t).sum(dims, dtype=torch.float64)
            s[2] += t.numel() // t.shape[1]
        return hook

    handles = [m.register_forward_pre_hook(make_hook(m)) for m in bns]
    model.eval()                 # dropout off
    for m in bns:
        m.train()                # BN uses batch statistics in this pass
    try:
        n = x_train.shape[0]
        for i in range(0, n, bs):
            xb = x_train[i:i + bs]
            if xb.shape[0] < 2:  # BN cannot normalise a single sample
                continue
            model(to_float(xb))
    finally:
        for h in handles:
            h.remove()
    for m in bns:
        s, ss, cnt = acc[m]
        mean = s / cnt
        var = (ss / cnt - mean * mean).clamp_min(0.0)
        m.running_mean.copy_(mean.to(m.running_mean.dtype))
        m.running_var.copy_(var.to(m.running_var.dtype))
    model.eval()


# ----------------------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------------------
def should_stop(losses, window: int, tol: float) -> bool:
    """[REPRO CHOICE] Stopping rule for "until the loss function ceased to improve" (p. 4).

    `losses` = full training cross-entropy (eval mode) after each epoch.  Stop when the relative
    improvement of the best loss within the last `window` epochs over the loss `window` epochs ago
    is below `tol`.  Test data is never consulted.

    NOTE: because this is a RELATIVE criterion, a loss that keeps shrinking from e.g. 1e-3 to 5e-4
    still counts as "improving", so with tiny training losses the --max-epochs cap may be the binding
    stopping condition.  That is reported at the end of every run."""
    if len(losses) <= window:
        return False
    ref = losses[-window - 1]
    best_recent = min(losses[-window:])
    rel_impr = (ref - best_recent) / max(abs(ref), 1e-12)
    return rel_impr < tol


def train_one(net, model, data, batch_size, seed, args, device, tag):
    model.to(device)
    # [REPRO CHOICE] ADAM defaults (Kingma & Ba): lr=1e-3, betas=(0.9,0.999), eps=1e-8, constant
    # schedule, no weight decay, and the SAME hyper-parameters for SB and LB ([INFERRED]: the paper
    # mentions no per-batch-size tuning).  [PAPER] optimizer = ADAM for both regimes (p. 4).
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
    torch.manual_seed(seed)                       # controls dropout masks
    g = torch.Generator().manual_seed(seed)       # controls batch order
    x, y = data["xtr"], data["ytr"]
    n = x.shape[0]
    hist = dict(train_loss=[], train_acc=[])
    stopped_by = "max_epochs"
    t0 = time.time()
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        # [REPRO CHOICE] reshuffle every epoch, sample WITHOUT replacement, keep the last partial
        # batch ([PAPER]: batches randomly sampled with fixed size, Sec. 1.1, p. 2).
        perm = torch.randperm(n, generator=g).to(device)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            if idx.numel() < 2:                    # BN cannot handle a batch of one
                continue
            loss = F.cross_entropy(model(to_float(x[idx])), y[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        # Stopping statistic: full TRAINING cross-entropy, eval mode.  (Test set is not touched.)
        tr_loss, tr_acc = evaluate(model, x, y, args.eval_bs)
        hist["train_loss"].append(tr_loss)
        hist["train_acc"].append(tr_acc)
        if not np.isfinite(tr_loss):
            raise RuntimeError(f"[{tag}] non-finite training loss at epoch {epoch}")
        if epoch % 5 == 0 or epoch == 1:
            log(f"  [{tag}] epoch {epoch:4d}  train loss {tr_loss:.6f}  train acc {tr_acc:6.2f}%  "
                f"({time.time() - t0:.0f}s)")
        if should_stop(hist["train_loss"], args.stop_window, args.stop_tol):
            stopped_by = "criterion"
            log(f"  [{tag}] stopping criterion met at epoch {epoch}")
            break
    # [INFERRED] the minimiser is the FINAL iterate (Fig. 4 caption, p. 8): no best-checkpoint selection.
    te_loss, te_acc = evaluate(model, data["xte"], data["yte"], args.eval_bs)
    final = dict(train_loss=float(hist["train_loss"][-1]), train_acc=float(hist["train_acc"][-1]),
                 test_loss=float(te_loss), test_acc=float(te_acc))
    log(f"  [{tag}] FINAL  train {final['train_loss']:.5f}/{final['train_acc']:.2f}%  "
        f"test {final['test_loss']:.4f}/{final['test_acc']:.2f}%  epochs={len(hist['train_loss'])} "
        f"stopped_by={stopped_by}")
    if stopped_by == "max_epochs":
        warn(f"[{tag}] hit --max-epochs={args.max_epochs} before the stopping criterion fired.")
    return dict(state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
                net=net, batch_size=int(batch_size), seed=int(seed), epochs=len(hist["train_loss"]),
                stopped_by=stopped_by, history=hist, final=final)


def get_or_train_pair(net, trial, data, lb_bs, args, device, ckpt_dir):
    p_sb = os.path.join(ckpt_dir, f"trial{trial}_sb.pt")
    p_lb = os.path.join(ckpt_dir, f"trial{trial}_lb.pt")
    if os.path.exists(p_sb) and os.path.exists(p_lb) and not args.retrain:
        log(f"  trial {trial}: checkpoints exist, skipping training (use --retrain to force)")
        return p_sb, p_lb

    init_seed = args.seed + 1000 * trial        # [REPRO CHOICE] seed scheme
    # SHARED INITIALISATION: build once, deep-copy for the second model.
    torch.manual_seed(init_seed)
    m_sb = build_model(net)
    init_params(m_sb)
    m_lb = copy.deepcopy(m_sb)
    for (k1, v1), (k2, v2) in zip(m_sb.state_dict().items(), m_lb.state_dict().items()):
        assert k1 == k2 and torch.equal(v1, v2), "SB/LB initial parameters differ"
    log(f"  trial {trial}: SB/LB share identical initial parameters (seed {init_seed}); "
        f"#params = {n_params(m_sb):,}")

    ck_sb = train_one(net, m_sb, data, args.sb_batch, init_seed + 1, args, device, f"{net} t{trial} SB bs={args.sb_batch}")
    ck_lb = train_one(net, m_lb, data, lb_bs, init_seed + 2, args, device, f"{net} t{trial} LB bs={lb_bs}")
    for ck in (ck_sb, ck_lb):
        ck["init_seed"] = int(init_seed)
        ck["args"] = {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                      for k, v in vars(args).items()}
    torch.save(ck_sb, p_sb)
    torch.save(ck_lb, p_lb)
    return p_sb, p_lb


def load_ckpt(path, device):
    return torch.load(path, map_location=device)


# ----------------------------------------------------------------------------------------
# Table-2-style sanity summary
# ----------------------------------------------------------------------------------------
def summarize_table2(net, paths, out_dir, device):
    rows = {"sb": [], "lb": []}
    stops = {"sb": 0, "lb": 0}
    for p_sb, p_lb in paths:
        for key, p in (("sb", p_sb), ("lb", p_lb)):
            ck = load_ckpt(p, "cpu")
            rows[key].append((ck["final"]["train_acc"], ck["final"]["test_acc"]))
            stops[key] += int(ck["stopped_by"] == "max_epochs")
    summ = {}
    for key in rows:
        a = np.array(rows[key])
        ddof = 1 if len(a) > 1 else 0         # [REPRO CHOICE] sample std
        summ[key] = dict(train_mean=a[:, 0].mean(), train_std=a[:, 0].std(ddof=ddof),
                         test_mean=a[:, 1].mean(), test_std=a[:, 1].std(ddof=ddof),
                         runs_hitting_epoch_cap=stops[key])
    log(f"\n  --- {net}: Table-2-style summary over {len(paths)} trial(s) ---")
    for key in ("sb", "lb"):
        s = summ[key]
        log(f"  {key.upper()}: train {s['train_mean']:.2f} +/- {s['train_std']:.2f}   "
            f"test {s['test_mean']:.2f} +/- {s['test_std']:.2f}   (runs at epoch cap: {s['runs_hitting_epoch_cap']})")
    # Sanity check 1: the paper's central qualitative claim (generalisation gap exists).
    gap = summ["sb"]["test_mean"] - summ["lb"]["test_mean"]
    log(f"  mean test-accuracy gap SB - LB = {gap:+.2f} pp")
    if gap <= 0:
        warn(f"{net}: no positive SB-LB generalisation gap observed; Figure 3 may not look as in the paper.")
    # Sanity check 2: very high final training accuracy for both regimes.
    for key in ("sb", "lb"):
        if summ[key]["train_mean"] < 95.0:
            warn(f"{net}: {key.upper()} mean training accuracy {summ[key]['train_mean']:.1f}% < 95% "
                 f"(not near a minimiser; consider raising --max-epochs).")
    # Sanity check 3 (informational only): compare with the paper's Table 2.
    if not getattr(summarize_table2, "synthetic", False) and net in PAPER_TABLE2_REF:
        ref = PAPER_TABLE2_REF[net]
        mine = (summ["sb"]["train_mean"], summ["sb"]["test_mean"], summ["lb"]["train_mean"], summ["lb"]["test_mean"])
        names = ("SB train", "SB test", "LB train", "LB test")
        for nme, r, m in zip(names, ref, mine):
            flag = "" if abs(r - m) <= TABLE2_TOL_PP else "   <-- differs by > %.1f pp" % TABLE2_TOL_PP
            log(f"  [info] {nme}: ours {m:.2f}  paper {r:.2f}{flag}")
    with open(os.path.join(out_dir, "table2_summary.json"), "w") as f:
        json.dump({k: {kk: float(vv) for kk, vv in v.items()} for k, v in summ.items()}, f, indent=2)
    return summ


# ----------------------------------------------------------------------------------------
# Interpolation
# ----------------------------------------------------------------------------------------
def make_alphas(n_alpha: int) -> np.ndarray:
    # [REPRO CHOICE] uniform grid on [-1, 2]; default 151 points (step 0.02); exact 0 and 1 included.
    a = np.round(np.linspace(ALPHA_MIN, ALPHA_MAX, n_alpha), 10)
    assert (a == 0.0).any() and (a == 1.0).any(), "alpha grid must contain exactly 0 and 1 " \
        "(use n_alpha = 3k+1, e.g. 31, 61, 151)"
    return a


def interpolate_pair(net, ck_sb, ck_lb, data, alphas, args, device):
    """theta(alpha) = alpha * x_LB + (1 - alpha) * x_SB, applied elementwise to every trainable
    parameter (weights, biases, BN gamma/beta) with the same alpha  [PAPER: Sec 2.2.1; INFERRED:
    x = all trainable parameters].  BN running statistics are handled according to --bn-mode."""
    model = build_model(net).to(device)
    pnames = set(dict(model.named_parameters()).keys())
    sb_sd = {k: v.to(device) for k, v in ck_sb["state_dict"].items()}
    lb_sd = {k: v.to(device) for k, v in ck_lb["state_dict"].items()}
    assert sb_sd.keys() == lb_sd.keys() == model.state_dict().keys()

    def mix(vs, vl, a):
        # float64 mixing then cast back: alpha=0 reproduces SB bit-exactly, alpha=1 reproduces LB.
        return (a * vl.double() + (1.0 - a) * vs.double()).float()

    # Sanity: endpoints of the path equal the trained models' parameters exactly.
    for k in pnames:
        assert torch.equal(mix(sb_sd[k], lb_sd[k], 0.0), sb_sd[k]), f"alpha=0 != SB for {k}"
        assert torch.equal(mix(sb_sd[k], lb_sd[k], 1.0), lb_sd[k]), f"alpha=1 != LB for {k}"

    res = {k: [] for k in ("train_loss", "train_acc", "test_loss", "test_acc")}
    t0 = time.time()
    for i, a in enumerate(alphas):
        a = float(a)
        sd = {}
        for k, vs in sb_sd.items():
            vl = lb_sd[k]
            if k in pnames:
                sd[k] = mix(vs, vl, a)
            elif vs.is_floating_point():                    # BN running_mean / running_var
                if args.bn_mode == "interp_buffers":
                    # Ablation (A): linearly interpolate the buffers too.  Extrapolated variances can
                    # become negative, so clamp at 0 (BN's eps keeps the division finite).
                    v = mix(vs, vl, a)
                    sd[k] = v.clamp_min(0.0) if k.endswith("running_var") else v
                else:
                    sd[k] = vs.clone()                      # placeholder, overwritten below
            else:
                sd[k] = vs.clone()                          # num_batches_tracked (unused in eval)
        model.load_state_dict(sd)
        if args.bn_mode == "reestimate":
            reestimate_bn(model, data["xtr"], args.bn_batch)
        trl, tra = evaluate(model, data["xtr"], data["ytr"], args.eval_bs)
        tel, tea = evaluate(model, data["xte"], data["yte"], args.eval_bs)
        for k, v in zip(res.keys(), (trl, tra, tel, tea)):
            res[k].append(v)
        if i % 10 == 0 or i == len(alphas) - 1:
            log(f"    alpha={a:+.2f}  train {trl:10.4f}/{tra:6.2f}%   test {tel:10.4f}/{tea:6.2f}%  "
                f"({time.time() - t0:.0f}s)")
    out = {k: np.array(v, dtype=np.float64) for k, v in res.items()}
    out["alphas"] = alphas.astype(np.float64)
    return out


def check_interpolation(net, res, ck_sb, ck_lb, args, delta=0.1):
    """Validation checks on the interpolation results (warnings, not hard failures)."""
    a = res["alphas"]
    i0, i1 = int(np.argmin(np.abs(a))), int(np.argmin(np.abs(a - 1.0)))
    ok = True
    # Check 1: endpoints reproduce the end-of-training numbers (computed with EMA running stats).
    for name, idx, ck in (("alpha=0 (SB)", i0, ck_sb), ("alpha=1 (LB)", i1, ck_lb)):
        f = ck["final"]
        for split in ("train", "test"):
            d_acc = abs(res[f"{split}_acc"][idx] - f[f"{split}_acc"])
            d_loss = abs(res[f"{split}_loss"][idx] - f[f"{split}_loss"])
            tol_loss = 0.05 + 0.10 * f[f"{split}_loss"]      # [REPRO CHOICE] tolerances
            if d_acc > 1.0 or d_loss > tol_loss:
                ok = False
                warn(f"{net} {name} {split}: interpolation endpoint differs from trained model "
                     f"(acc diff {d_acc:.2f} pp, loss diff {d_loss:.4f}); BN re-estimation may differ "
                     f"from the EMA statistics, or something is off.")
    log(f"  [check] endpoints match end-of-training metrics: {'OK' if ok else 'MISMATCH (see warnings)'}")
    # Check 2: no NaN/inf in the path.
    for k in ("train_loss", "test_loss"):
        if not np.all(np.isfinite(res[k])):
            warn(f"{net}: non-finite values in {k} along the path (kept as-is in the raw results).")
    # Check 3 (diagnostics): paper's qualitative claim = loss rises more steeply around the LB end.
    def rise(center_idx, center_alpha):
        vals = []
        for s in (-delta, +delta):
            j = int(np.argmin(np.abs(a - (center_alpha + s))))
            vals.append(res["train_loss"][j] - res["train_loss"][center_idx])
        return float(np.mean(vals))
    r_sb, r_lb = rise(i0, 0.0), rise(i1, 1.0)
    log(f"  [diag] mean rise of TRAIN loss at alpha = center +/- {delta}:  SB end {r_sb:.4f}   LB end {r_lb:.4f}   "
        f"(paper's claim: LB end rises more steeply)")
    log(f"  [diag] test accuracy at alpha=0: {res['test_acc'][i0]:.2f}%   at alpha=1: {res['test_acc'][i1]:.2f}%")
    if r_lb <= r_sb:
        warn(f"{net}: LB end is not steeper than SB end for this pair (single-pair outcome; "
             f"inspect other trials with --pairs all).")
    return dict(endpoints_ok=bool(ok), rise_sb=r_sb, rise_lb=r_lb,
                test_acc_alpha0=float(res["test_acc"][i0]), test_acc_alpha1=float(res["test_acc"][i1]))


# ----------------------------------------------------------------------------------------
# Saving / plotting
# ----------------------------------------------------------------------------------------
def save_raw(res, meta, base):
    np.savez(base + ".npz", **res)
    with open(base + ".csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["alpha", "train_loss", "train_acc", "test_loss", "test_acc"])
        for row in zip(res["alphas"], res["train_loss"], res["train_acc"], res["test_loss"], res["test_acc"]):
            w.writerow([f"{v:.10g}" for v in row])
    with open(base + "_meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)


def plot_fig3(res, net, panel, base, title_extra=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    a = res["alphas"]
    fig, ax1 = plt.subplots(figsize=(5.4, 4.3))
    ax2 = ax1.twinx()
    # Same structure as Fig. 3: left axis = cross-entropy (blue), right axis = accuracy (red);
    # solid = training set, dashed = testing set; alpha=0 is SB, alpha=1 is LB.
    ax1.plot(a, res["train_loss"], color="tab:blue", ls="-")
    ax1.plot(a, res["test_loss"], color="tab:blue", ls="--")
    ax2.plot(a, res["train_acc"], color="tab:red", ls="-")
    ax2.plot(a, res["test_acc"], color="tab:red", ls="--")
    ax1.set_xlabel("alpha")
    ax1.set_ylabel("Cross Entropy", color="tab:blue")
    ax2.set_ylabel("Accuracy", color="tab:red")
    ax1.set_xlim(ALPHA_MIN, ALPHA_MAX)
    ax1.set_ylim(bottom=0)
    ax2.set_ylim(0, 100)
    ax1.grid(True, ls=":")
    ax1.legend(handles=[Line2D([0], [0], color="k", ls="-", label="Train"),
                        Line2D([0], [0], color="k", ls="--", label="Test")], loc="upper left")
    ax1.set_title(f"({panel[-1]}) {net}  {title_extra}".strip(), fontsize=10)
    fig.tight_layout()
    fig.savefig(base + ".png", dpi=200)
    fig.savefig(base + ".pdf")
    plt.close(fig)


# ----------------------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------------------
def run_net(net, args, device):
    cfg = NET_CONFIGS[net]
    out_dir = os.path.join(args.out_dir, net)
    ckpt_dir = os.path.join(out_dir, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    log(f"\n==================== {net}  ({cfg['dataset']}, Figure {cfg['panel']}) ====================")

    data = load_data(cfg["dataset"], args.data_dir, device, args.synthetic, args.seed)
    n_train = data["xtr"].shape[0]
    # [PAPER] LB batch = 10% of the training set, SB batch = 256 (p. 4; App. E, p. 13).
    lb_bs = int(round(args.lb_frac * n_train))
    log(f"  train/test = {n_train}/{data['xte'].shape[0]};  SB batch = {args.sb_batch};  LB batch = {lb_bs}")
    if args.synthetic == 0:
        assert lb_bs == {"mnist": 6000, "cifar10": 5000}[cfg["dataset"]]
    if lb_bs <= args.sb_batch:
        warn("LB batch size is not larger than SB batch size (only expected in --synthetic smoke tests).")

    # --- model sanity: shapes, eval determinism -------------------------------------------------
    m = build_model(net).to(device)
    log(f"  architecture sanity: #params = {n_params(m):,}")
    with torch.no_grad():
        m.eval()
        o = m(to_float(data["xtr"][:4]))
        assert o.shape == (4, 10), f"unexpected output shape {tuple(o.shape)}"
    l1 = evaluate(m, data["xte"], data["yte"], args.eval_bs)
    l2 = evaluate(m, data["xte"], data["yte"], args.eval_bs)
    assert l1 == l2, "evaluation is not deterministic (dropout active in eval?)"
    del m

    # --- stage: train ---------------------------------------------------------------------------
    paths = []
    if args.stage in ("train", "all"):
        for t in range(args.trials):
            log(f"\n  >>> trial {t + 1}/{args.trials}")
            paths.append(get_or_train_pair(net, t, data, lb_bs, args, device, ckpt_dir))
        summarize_table2.synthetic = args.synthetic > 0
        summarize_table2(net, paths, out_dir, device)
    else:
        paths = [(os.path.join(ckpt_dir, f"trial{t}_sb.pt"), os.path.join(ckpt_dir, f"trial{t}_lb.pt"))
                 for t in range(args.trials)]

    # --- pair selection -------------------------------------------------------------------------
    # [PAPER] a pair was "randomly chosen" from the 5 trials (p. 5).
    # [REPRO CHOICE] pre-declared seeded draw that does not depend on any results.
    chosen = int(np.random.RandomState(args.seed).randint(args.trials))
    trials_to_use = list(range(args.trials)) if args.pairs == "all" else [chosen]
    log(f"\n  pair selection: seed {args.seed} -> trial {chosen} (primary figure); interpolating trials {trials_to_use}")

    alphas = make_alphas(args.n_alpha)
    tag = args.bn_mode
    for t in trials_to_use:
        base = os.path.join(out_dir, f"interp_trial{t}_{tag}")
        if args.stage in ("interpolate", "all"):
            p_sb, p_lb = paths[t]
            for p in (p_sb, p_lb):
                if not os.path.exists(p):
                    raise FileNotFoundError(f"{p} missing; run with --stage train (or all) first")
            ck_sb, ck_lb = load_ckpt(p_sb, device), load_ckpt(p_lb, device)
            log(f"\n  >>> interpolating trial {t} (bn-mode = {args.bn_mode}, {len(alphas)} points)")
            res = interpolate_pair(net, ck_sb, ck_lb, data, alphas, args, device)
            diag = check_interpolation(net, res, ck_sb, ck_lb, args)
            meta = dict(net=net, panel=cfg["panel"], dataset=cfg["dataset"], trial=t, chosen_trial=chosen,
                        sb_batch=args.sb_batch, lb_batch=lb_bs, init_seed=ck_sb["init_seed"],
                        sb_epochs=ck_sb["epochs"], lb_epochs=ck_lb["epochs"],
                        sb_stopped_by=ck_sb["stopped_by"], lb_stopped_by=ck_lb["stopped_by"],
                        sb_final=ck_sb["final"], lb_final=ck_lb["final"], diagnostics=diag,
                        bn_mode=args.bn_mode, n_alpha=len(alphas), args=vars(args),
                        torch=torch.__version__, numpy=np.__version__, python=sys.version,
                        platform=platform.platform(), synthetic=args.synthetic > 0)
            save_raw(res, meta, base)
        if args.stage in ("plot", "all", "interpolate"):
            if not os.path.exists(base + ".npz"):
                warn(f"no raw results at {base}.npz; run --stage interpolate first")
                continue
            res = dict(np.load(base + ".npz"))
            suffix = "" if t == chosen else f" (trial {t})"
            fname = f"fig{cfg['panel']}_{net}_trial{t}_{tag}"
            plot_fig3(res, net, cfg["panel"], os.path.join(out_dir, fname),
                      title_extra=f"[trial {t}, bn={tag}]" + suffix)
            log(f"  saved figure: {os.path.join(out_dir, fname)}.png")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nets", nargs="+", default=["F1", "C1", "C2"], choices=list(NET_CONFIGS))
    p.add_argument("--stage", default="all", choices=["train", "interpolate", "plot", "all"])
    p.add_argument("--trials", type=int, default=5, help="[PAPER] 5 runs (p. 4)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="results")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--device", default="auto")
    p.add_argument("--retrain", action="store_true", help="ignore existing checkpoints")
    # training
    p.add_argument("--sb-batch", type=int, default=256, help="[PAPER] 256")
    p.add_argument("--lb-frac", type=float, default=0.10, help="[PAPER] 10%% of the training set")
    p.add_argument("--lr", type=float, default=1e-3, help="[REPRO CHOICE] ADAM default")
    p.add_argument("--max-epochs", type=int, default=200, help="[REPRO CHOICE] cap")
    p.add_argument("--stop-window", type=int, default=20, help="[REPRO CHOICE]")
    p.add_argument("--stop-tol", type=float, default=1e-3, help="[REPRO CHOICE] relative improvement")
    # interpolation
    p.add_argument("--n-alpha", type=int, default=151, help="[REPRO CHOICE] must be 3k+1 so 0 and 1 are on the grid")
    p.add_argument("--bn-mode", default="reestimate", choices=["reestimate", "interp_buffers"],
                   help="reestimate = primary (B); interp_buffers = ablation (A)")
    p.add_argument("--pairs", default="chosen", choices=["chosen", "all"])
    p.add_argument("--eval-bs", type=int, default=1000, help="evaluation chunk size (no effect on results)")
    p.add_argument("--bn-batch", type=int, default=500, help="[REPRO CHOICE] batch size for the BN re-estimation pass")
    p.add_argument("--synthetic", type=int, default=0,
                   help="N>0: use N random training samples (pipeline smoke test only)")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else
                          ("cpu" if args.device == "auto" else args.device))
    set_determinism(args.seed)
    log(f"device = {device}; torch {torch.__version__}; seed = {args.seed}")
    os.makedirs(args.out_dir, exist_ok=True)
    for net in args.nets:
        set_determinism(args.seed)
        run_net(net, args, device)
    log("\nDone.")


if __name__ == "__main__":
    main()
