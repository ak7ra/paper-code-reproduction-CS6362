#!/usr/bin/env python3
"""
Reproduction of Figure 1(a-c) of Zhang et al., "Understanding deep learning
requires rethinking generalization" (ICLR 2017).

Provenance tags used in comments:
  [PAPER]     explicitly stated in the paper (page cited)
  [INFERRED]  derived from the paper's text / plots / tables
  [PROPOSED]  paper is silent; this is OUR choice and must be reported as ours

Subcommands:
  validate   build checks (param counts, preprocessing, corruption invariants)
  sweep      label-corruption sweep (Fig 1b, 1c; also the true/random-label
             curves of Fig 1a, which are the Inception p=0 and p=1 cells)
  fig1a      the three input-corruption runs of Fig 1a (Inception only)
  aggregate  write CSVs with the data for Fig 1a, 1b, 1c

Example:
  python reproduce_fig1.py validate
  python reproduce_fig1.py sweep  --tier minimal --out results
  python reproduce_fig1.py fig1a  --tier minimal --out results
  python reproduce_fig1.py aggregate --out results
"""
import argparse
import csv
import glob
import json
import math
import os
import time
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
NUM_CLASSES = 10
CROP = 28                      # [PAPER p.12] center crop 32x32 -> 28x28
MOMENTUM = 0.9                 # [PAPER p.12]
LR0 = {"inception": 0.1,       # [PAPER p.12] 0.1 for small Inception
       "alexnet": 0.01,        # [PAPER p.12] 0.01 for small AlexNet and MLPs
       "mlp1x512": 0.01}
LR_DECAY = 0.95                # [PAPER p.12] decay factor 0.95 per training epoch
# [PAPER p.6, Table 1] parameter counts, used as build validation
PARAM_COUNTS = {"inception": 1_649_402, "alexnet": 1_387_786, "mlp1x512": 1_209_866}

# Seed salts so that label draws, input-corruption draws and data order use
# independent generators derived from the single run seed. [PROPOSED]
SALT_LABEL, SALT_INPUT, SALT_ORDER = 10_000_019, 20_000_003, 30_000_001

# ---- Experiment grids ------------------------------------------------------
P_FULL = [round(i / 10, 2) for i in range(11)]          # [PROPOSED] 11 levels
P_REDUCED = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]              # [PROPOSED]
P_MINIMAL = [0.0, 0.5, 1.0]                             # [PROPOSED]
ARCHS = ["inception", "alexnet", "mlp1x512"]
TIERS = {  # [PROPOSED] reduction tiers from the plan (seeds are 0..k-1)
    "full": dict(p=P_FULL, seeds={"inception": 3, "alexnet": 3, "mlp1x512": 3}, fig1a_seeds=3),
    "reduced": dict(p=P_REDUCED, seeds={"inception": 2, "alexnet": 3, "mlp1x512": 3}, fig1a_seeds=2),
    "minimal": dict(p=P_MINIMAL, seeds={"inception": 1, "alexnet": 1, "mlp1x512": 1}, fig1a_seeds=1),
}
BUDGET_1B1C = 40_000   # [PROPOSED] steps, label-corruption sweep, no early stopping
BUDGET_1A = 30_000     # [PROPOSED] steps for Fig 1a runs (curves truncated to this)
INPUT_SETTINGS = ["shuffled_pixels", "random_pixels", "gaussian"]


@dataclass
class RunConfig:
    arch: str
    setting: str = "label_corruption"  # | shuffled_pixels | random_pixels | gaussian
    p: float = 0.0                     # label corruption prob (0 for input settings)
    seed: int = 0
    budget_steps: int = BUDGET_1B1C
    batch_size: int = 128              # [PROPOSED] paper does not state batch size
    eval_every: int = 200              # [PROPOSED] full-pass eval every 200 steps
    # Convergence criterion (configurable). [PROPOSED] default = 99% train acc,
    # held at two consecutive eval points. Paper only says "fit perfectly" (p.5).
    conv_metric: str = "train_acc"     # "train_acc" or "train_loss"
    conv_threshold: float = 0.99
    conv_consecutive: int = 2
    data_root: str = "./data"


# --------------------------------------------------------------------------
# Data: preprocessing, input corruption, label corruption
# --------------------------------------------------------------------------
def per_image_standardize(x, chunk=5000):
    """[PAPER p.12] per-image standardization (TF per_image_whitening).
    The exact formula is NOT given in the paper. [PROPOSED]: subtract the
    per-image mean, divide by max(population std, 1/sqrt(N)), N = C*H*W,
    channels pooled. Returns (standardized tensor, number of floor hits)."""
    n = x[0].numel()
    floor = 1.0 / math.sqrt(n)
    out = torch.empty_like(x)
    hits = 0
    for i in range(0, len(x), chunk):
        xb = x[i:i + chunk]
        mean = xb.mean(dim=(1, 2, 3), keepdim=True)
        std = xb.std(dim=(1, 2, 3), keepdim=True, correction=0)  # population std
        hits += int((std < floor).sum())
        out[i:i + chunk] = (xb - mean) / std.clamp_min(floor)
    return out, hits


class BaseData:
    """CIFAR10 after /255 and 28x28 center crop (before any corruption)."""

    def __init__(self, root, device):
        from torchvision.datasets import CIFAR10
        tr = CIFAR10(root, train=True, download=True)
        te = CIFAR10(root, train=False, download=True)
        self.device = device
        self.train_raw = self._prep(tr.data)          # CPU float in [0,1], (N,3,28,28)
        self.test_raw = self._prep(te.data)
        self.train_y = torch.tensor(tr.targets, dtype=torch.long)
        self.test_y = torch.tensor(te.targets, dtype=torch.long)
        # Clean standardized copies, cached on device (used by label_corruption)
        self.train_std, self.train_hits = per_image_standardize(self.train_raw.to(device))
        self.test_std, self.test_hits = per_image_standardize(self.test_raw.to(device))

    @staticmethod
    def _prep(arr):
        x = torch.from_numpy(arr).permute(0, 3, 1, 2).float().div_(255.0)  # [PAPER p.12]
        o = (32 - CROP) // 2                                               # center crop
        return x[:, :, o:o + CROP, o:o + CROP].contiguous()


def make_label_draws(n, seed):
    """Corruption randomness, generated ONCE per seed. [PROPOSED]
    u_i ~ U[0,1) and r_i ~ Uniform{0..9} (all 10 classes, incl. the true one;
    literal reading of 'uniform random class', p.4 -> realized flip rate ~0.9p).
    The SAME (u, r) are reused for every p within a seed (nested corruption) and
    for every architecture (paired comparisons)."""
    g = torch.Generator().manual_seed(seed + SALT_LABEL)
    u = torch.rand(n, generator=g)
    r = torch.randint(0, NUM_CLASSES, (n,), generator=g)
    return u, r


def corrupt_labels(y_true, u, r, p):
    """[PAPER p.4] independently with probability p, replace the label with a
    uniform random class. p=0 -> no change, p=1 -> all labels random.
    Returns a NEW tensor; inputs are never modified in place."""
    mask = u < p
    y = y_true.clone()
    y[mask] = r[mask]
    return y


def spatial_permute(x, idx):
    """Permute the 784 spatial locations of (N,3,28,28), RGB triplets move
    together. [PROPOSED] ('pixel' = spatial location). idx is (784,) or (N,784)."""
    n = x.shape[0]
    flat = x.reshape(n, 3, CROP * CROP)
    if idx.dim() == 1:
        out = flat[:, :, idx]
    else:
        out = flat.gather(2, idx.unsqueeze(1).expand(-1, 3, -1))
    return out.reshape(n, 3, CROP, CROP)


def build_inputs(base, setting, seed):
    """Returns (train_x, test_x_or_None) on device, fully preprocessed.
    [PROPOSED] Order: crop -> corrupt -> per-image standardize. All corruption
    is generated once here, stored, and never regenerated during training."""
    dev = base.device
    if setting == "label_corruption":
        return base.train_std, base.test_std
    g = torch.Generator().manual_seed(seed + SALT_INPUT)
    n = base.train_raw.shape[0]
    if setting == "shuffled_pixels":
        # [PAPER p.4] one fixed permutation applied to ALL train and test images
        perm = torch.randperm(CROP * CROP, generator=g)
        tr = spatial_permute(base.train_raw, perm)
        te = spatial_permute(base.test_raw, perm)
        return (per_image_standardize(tr.to(dev))[0], per_image_standardize(te.to(dev))[0])
    if setting == "random_pixels":
        # [PAPER p.4] a different permutation per image (train only; Fig 1a shows
        # training loss only, so no test set is built [PROPOSED])
        idx = torch.rand(n, CROP * CROP, generator=g).argsort(dim=1)
        tr = spatial_permute(base.train_raw, idx)
        return per_image_standardize(tr.to(dev))[0], None
    if setting == "gaussian":
        # [PAPER p.4] Gaussian 'with matching mean and variance'. [PROPOSED]: one
        # global scalar mean/std from the cropped [0,1] training pixels; iid
        # samples for every location of every training image, drawn once.
        mu, sigma = base.train_raw.mean().item(), base.train_raw.std().item()
        tr = mu + sigma * torch.randn(n, 3, CROP, CROP, generator=g)
        return per_image_standardize(tr.to(dev))[0], None
    raise ValueError(setting)


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
class ConvModule(nn.Module):
    """[PAPER p.12, Fig 3] Convolution -> BatchNorm -> ReLU.
    [INFERRED] conv has NO bias and BatchNorm has a learned shift (beta) but NO
    learned scale (gamma): this is what reproduces Table 1's 1,649,402 params
    (hand arithmetic: bias-free conv + full affine BN is 1,712 params too many,
    and 1,712 = sum of conv output channels). Padding 'same' [PROPOSED]."""

    def __init__(self, cin, cout, k, stride):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, stride, padding=k // 2, bias=False)
        self.bn = nn.BatchNorm2d(cout, affine=False)   # eps/momentum: PyTorch defaults [PROPOSED]
        self.beta = nn.Parameter(torch.zeros(cout))

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)) + self.beta.view(1, -1, 1, 1))


class InceptionModule(nn.Module):  # [PAPER Fig 3] 1x1 branch + 3x3 branch, concat
    def __init__(self, cin, ch1, ch3):
        super().__init__()
        self.b1 = ConvModule(cin, ch1, 1, 1)
        self.b3 = ConvModule(cin, ch3, 3, 1)

    def forward(self, x):
        return torch.cat([self.b1(x), self.b3(x)], dim=1)


class DownsampleModule(nn.Module):  # [PAPER Fig 3] strided 3x3 conv + 3x3/2 maxpool, concat
    def __init__(self, cin, ch3):
        super().__init__()
        self.conv = ConvModule(cin, ch3, 3, 2)
        self.pool = nn.MaxPool2d(3, 2, padding=1)  # padding 1 -> 28->14->7 [INFERRED]

    def forward(self, x):
        return torch.cat([self.conv(x), self.pool(x)], dim=1)


class SmallInception(nn.Module):
    """[PAPER Fig 3, p.12]. Module sequence read from the figure:
    Conv96 -> I(32+32) -> I(32+48) -> Down80 -> I(112+48) -> I(96+64) ->
    I(80+80) -> I(48+96) -> Down96 -> I(176+160) -> I(176+160) -> 7x7 mean pool -> FC10"""

    def __init__(self):
        super().__init__()
        layers, c = [ConvModule(3, 96, 3, 1)], 96
        for ch1, ch3 in [(32, 32), (32, 48)]:
            layers.append(InceptionModule(c, ch1, ch3)); c = ch1 + ch3
        layers.append(DownsampleModule(c, 80)); c = c + 80
        for ch1, ch3 in [(112, 48), (96, 64), (80, 80), (48, 96)]:
            layers.append(InceptionModule(c, ch1, ch3)); c = ch1 + ch3
        layers.append(DownsampleModule(c, 96)); c = c + 96
        for ch1, ch3 in [(176, 160), (176, 160)]:
            layers.append(InceptionModule(c, ch1, ch3)); c = ch1 + ch3
        self.features = nn.Sequential(*layers)
        self.fc = nn.Linear(c, NUM_CLASSES)

    def forward(self, x):
        x = self.features(x)
        return self.fc(F.avg_pool2d(x, 7).flatten(1))


class SmallAlexNet(nn.Module):
    """[PAPER p.12] two (conv5x5 -> max-pool 3x3 -> LRN) modules, FC-384, FC-192,
    10-way linear. Filter counts / strides / padding / LRN constants are NOT
    given. [INFERRED] 64 filters, 'same' padding, stride-2 pooling reproduce
    Table 1's 1,387,786 params. [PROPOSED] ReLU after conv and before pool;
    LRN constants (size 9, alpha 1e-3, beta 0.75, k 1.0) are our choice, from
    background knowledge, not from the paper."""

    def __init__(self):
        super().__init__()
        def block(cin):
            return nn.Sequential(nn.Conv2d(cin, 64, 5, padding=2), nn.ReLU(),
                                 nn.MaxPool2d(3, 2, padding=1),
                                 nn.LocalResponseNorm(9, alpha=1e-3, beta=0.75, k=1.0))
        self.features = nn.Sequential(block(3), block(64))   # 28 -> 14 -> 7
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(7 * 7 * 64, 384), nn.ReLU(),
            nn.Linear(384, 192), nn.ReLU(), nn.Linear(192, NUM_CLASSES))

    def forward(self, x):
        return self.classifier(self.features(x))


class MLP1x512(nn.Module):  # [PAPER p.12] one hidden layer, 512 ReLU units
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(CROP * CROP * 3, 512),
                                 nn.ReLU(), nn.Linear(512, NUM_CLASSES))

    def forward(self, x):
        return self.net(x)


def build_model(arch):
    return {"inception": SmallInception, "alexnet": SmallAlexNet, "mlp1x512": MLP1x512}[arch]()


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# --------------------------------------------------------------------------
# Evaluation, convergence, training
# --------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, x, y, bs=1000):
    """Full-pass, EVAL-mode (BN running stats) cross-entropy and accuracy. [PROPOSED]"""
    model.eval()
    loss = torch.zeros((), device=x.device)
    correct = torch.zeros((), device=x.device)
    for i in range(0, len(x), bs):
        out = model(x[i:i + bs])
        loss += F.cross_entropy(out, y[i:i + bs], reduction="sum")
        correct += (out.argmax(1) == y[i:i + bs]).sum()
    model.train()
    return loss.item() / len(x), correct.item() / len(x)


def convergence_step(log, metric="train_acc", threshold=0.99, consecutive=2):
    """First eval step at which the criterion holds at `consecutive` successive
    eval points (default: that point and the next). Returns None if censored.
    Works on logged curves, so alternative thresholds can be applied post hoc."""
    ok = []
    for r in log:
        v = r[metric]
        ok.append(v is not None and (v >= threshold if metric == "train_acc" else v <= threshold))
    for i in range(len(log) - consecutive + 1):
        if all(ok[i:i + consecutive]):
            return log[i]["step"]
    return None


def train_run(cfg, base):
    dev = base.device
    train_x, test_x = build_inputs(base, cfg.setting, cfg.seed)
    y_true = base.train_y
    if cfg.setting == "label_corruption":
        u, r = make_label_draws(len(y_true), cfg.seed)
        train_y = corrupt_labels(y_true, u, r, cfg.p).to(dev)   # generated ONCE
    else:
        assert cfg.p == 0.0, "input-corruption settings use true labels"
        train_y = y_true.clone().to(dev)
    test_y = base.test_y.to(dev)
    y_snapshot = train_y.clone()                                 # for the fixed-label check
    changed_frac = (train_y.cpu() != y_true).float().mean().item()
    mask_frac = (cfg.p if cfg.setting == "label_corruption" else 0.0)

    torch.manual_seed(cfg.seed)            # init seed [PROPOSED]; PyTorch default init
    model = build_model(cfg.arch).to(dev)
    n_params = count_params(model)
    lr0 = LR0[cfg.arch]
    opt = torch.optim.SGD(model.parameters(), lr=lr0, momentum=MOMENTUM)  # [PAPER p.12]
    loss_fn = nn.CrossEntropyLoss()        # [PROPOSED] softmax cross-entropy
    gen = torch.Generator().manual_seed(cfg.seed + SALT_ORDER)

    N = len(train_x)
    steps_per_epoch = math.ceil(N / cfg.batch_size)   # drop_last=False [PROPOSED]
    assert cfg.budget_steps % cfg.eval_every == 0

    log, win_loss, win_correct, win_n = [], 0.0, 0.0, 0
    win_loss_t = torch.zeros((), device=dev)
    win_corr_t = torch.zeros((), device=dev)
    diverged = False
    converged = False

    def record(step, lr):
        nonlocal win_loss_t, win_corr_t, win_n
        tl, ta = evaluate(model, train_x, train_y)
        te = None
        if test_x is not None:
            _, tacc = evaluate(model, test_x, test_y)
            te = 1.0 - tacc                                  # top-1 test error
        rec = dict(step=step, epoch=step / steps_per_epoch, lr=lr,
                   train_loss=tl, train_acc=ta, test_err=te,
                   # secondary: train-mode running stats over the last window
                   run_loss=(win_loss_t.item() / win_n) if win_n else None,
                   run_acc=(win_corr_t.item() / win_n) if win_n else None)
        log.append(rec)
        win_loss_t.zero_(); win_corr_t.zero_(); win_n = 0
        if step % 1000 == 0:
            print(f"  [{cfg.arch} {cfg.setting} p={cfg.p} s={cfg.seed}] step {step} "
                  f"loss {tl:.4f} acc {ta:.4f} test_err {te}", flush=True)
        return math.isfinite(tl)

    step, t0 = 0, time.time()
    model.train()
    record(0, lr0)
    while step < cfg.budget_steps and not diverged and not converged:
        epoch = step // steps_per_epoch
        lr = lr0 * LR_DECAY ** epoch         # [PAPER 0.95/epoch]; staircase [PROPOSED]
        for g_ in opt.param_groups:
            g_["lr"] = lr
        perm = torch.randperm(N, generator=gen).to(dev)   # reshuffle each epoch [PROPOSED]
        for i in range(0, N, cfg.batch_size):
            if step >= cfg.budget_steps:
                break
            idx = perm[i:i + cfg.batch_size]
            out = model(train_x[idx])
            loss = loss_fn(out, train_y[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            b = len(idx)
            win_loss_t += loss.detach() * b
            win_corr_t += (out.argmax(1) == train_y[idx]).sum()
            win_n += b
            if step % cfg.eval_every == 0:
                if not record(step, lr):
                    diverged = True
                    break

                conv = convergence_step(log)
                if conv is not None:
                    print(
                        f"Converged at step {conv}; stopping early.",
                        flush=True
                    )
                    converged = True
                    break

    # Corrupted labels must be identical to what they were at the start.
    assert torch.equal(train_y, y_snapshot), "training labels changed during training!"

    conv = convergence_step(log, cfg.conv_metric, cfg.conv_threshold, cfg.conv_consecutive)
    final_test_err = log[-1]["test_err"]
    return dict(config=asdict(cfg), n_params=n_params, steps_per_epoch=steps_per_epoch,
                realized_changed_frac=changed_frac, label_mask_frac=mask_frac,
                conv_step=conv, conv_epoch=(conv / steps_per_epoch if conv is not None else None),
                censored=conv is None, final_test_err=final_test_err,
                diverged=diverged, elapsed_sec=time.time() - t0, log=log)


def run_path(out, cfg):
    bs = "" if cfg.batch_size == 128 else f"_bs{cfg.batch_size}"
    return os.path.join(out, "runs", f"{cfg.arch}_{cfg.setting}_p{cfg.p:.2f}_s{cfg.seed}{bs}.json")


def run_many(cfgs, base, out):
    os.makedirs(os.path.join(out, "runs"), exist_ok=True)
    for cfg in cfgs:
        path = run_path(out, cfg)
        if os.path.exists(path):
            print("skip (exists):", path); continue
        res = train_run(cfg, base)
        with open(path, "w") as f:
            json.dump(res, f)
        print(f"DONE {path}: conv_step={res['conv_step']} final_test_err={res['final_test_err']}")


# --------------------------------------------------------------------------
# Validation (build gate)
# --------------------------------------------------------------------------
def cmd_validate(args):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for a in ARCHS:                                   # [PAPER Table 1] param counts
        n = count_params(build_model(a))
        print(f"{a}: {n} params (expected {PARAM_COUNTS[a]})")
        assert n == PARAM_COUNTS[a], f"parameter count mismatch for {a}"
        out = build_model(a)(torch.randn(2, 3, 28, 28))
        assert out.shape == (2, 10)
    base = BaseData(args.data_root, dev)
    x = base.train_std
    m = x.mean(dim=(1, 2, 3)); s = x.std(dim=(1, 2, 3), correction=0)
    print(f"standardization: mean|max|={m.abs().max():.2e}, std range=[{s.min():.4f},{s.max():.4f}], "
          f"floor hits train={base.train_hits} test={base.test_hits}")
    assert m.abs().max() < 1e-4
    y = base.train_y; n = len(y)
    u, r = make_label_draws(n, 0)
    u2, r2 = make_label_draws(n, 0)
    assert torch.equal(u, u2) and torch.equal(r, r2), "draws not deterministic"
    prev = None
    for p in [0.0, 0.2, 0.5, 1.0]:
        yc = corrupt_labels(y, u, r, p)
        ch = (yc != y)
        print(f"p={p}: mask frac {(u < p).float().mean():.4f}, realized changed {ch.float().mean():.4f} "
              f"(expect ~{0.9 * p:.3f})")
        assert abs(ch.float().mean().item() - 0.9 * p) < 0.01
        if prev is not None:
            assert (ch | ~prev).all(), "corruption not nested"   # prev changed => now changed
        prev = ch
    assert torch.equal(corrupt_labels(y, u, r, 0.0), y)
    for st in INPUT_SETTINGS:
        a, b = build_inputs(base, st, 0)
        a2, _ = build_inputs(base, st, 0)
        assert torch.equal(a, a2), f"{st} not deterministic"
        print(st, tuple(a.shape), "test:", None if b is None else tuple(b.shape))
    print("ALL VALIDATION CHECKS PASSED")


# --------------------------------------------------------------------------
# Sweep drivers
# --------------------------------------------------------------------------
def conv_kwargs(args):
    return dict(conv_metric=args.metric, conv_threshold=args.threshold,
                conv_consecutive=args.consecutive, batch_size=args.batch_size,
                data_root=args.data_root)


def cmd_sweep(args):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True  # faster; runs not bitwise reproducible
    tier = TIERS[args.tier]
    ps = args.p if args.p else tier["p"]
    base = BaseData(args.data_root, dev)
    cfgs = []
    for arch in (args.archs or ARCHS):
        nseeds = args.seeds if args.seeds else tier["seeds"][arch]
        for seed in range(nseeds):
            for p in ps:
                cfgs.append(RunConfig(arch=arch, p=p, seed=seed, budget_steps=args.budget,
                                      **conv_kwargs(args)))
    print(f"{len(cfgs)} runs")
    run_many(cfgs, base, args.out)


def cmd_fig1a(args):
    """Input-corruption runs for Fig 1a (Inception only). True-label and
    random-label curves come from the sweep's Inception p=0 and p=1 runs."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    base = BaseData(args.data_root, dev)
    nseeds = args.seeds if args.seeds else TIERS[args.tier]["fig1a_seeds"]
    cfgs = [RunConfig(arch="inception", setting=st, p=0.0, seed=s, budget_steps=BUDGET_1A,
                      **conv_kwargs(args))
            for s in range(nseeds) for st in INPUT_SETTINGS]
    run_many(cfgs, base, args.out)


# --------------------------------------------------------------------------
# Aggregation -> CSVs for Fig 1a / 1b / 1c
# --------------------------------------------------------------------------
TABLE1_REF = {  # test error implied by Table 1 (100 - test acc), unregularized rows
    "inception": 0.1425, "alexnet": 0.2393, "mlp1x512": 0.4949}  # at p=0; ~0.90 at p=1


def write_csv(path, rows):
    if not rows:
        print("no rows for", path); return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("wrote", path, f"({len(rows)} rows)")


def cmd_aggregate(args):
    runs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(args.out, "runs", "*.json")))]
    runs = [r for r in runs if r["config"]["batch_size"] == args.batch_size]
    lab = [r for r in runs if r["config"]["setting"] == "label_corruption"]

    # ---- Fig 1a: training loss vs steps for 5 settings (Inception) -------
    rows = []
    for r in runs:
        c = r["config"]
        if c["arch"] != "inception":
            continue
        if c["setting"] == "label_corruption":
            if c["p"] == 0.0: name = "true_labels"
            elif c["p"] == 1.0: name = "random_labels"
            else: continue
        else:
            name = c["setting"]
        for rec in r["log"]:
            if rec["step"] <= BUDGET_1A:    # truncate 40k label runs to the 1a range
                rows.append(dict(setting=name, seed=c["seed"], step=rec["step"], epoch=rec["epoch"],
                                 train_loss_eval=rec["train_loss"], train_acc_eval=rec["train_acc"],
                                 train_loss_running=rec["run_loss"]))
    write_csv(os.path.join(args.out, "fig1a.csv"), rows)

    # ---- Fig 1b: relative convergence time -------------------------------
    criteria = [(args.metric, args.threshold, args.consecutive)]          # primary
    for th in (0.95, 0.999):                                              # pre-registered sensitivity
        criteria.append(("train_acc", th, args.consecutive))
    if args.loss_threshold is not None:   # plan says "fixed small value" without a number
        criteria.append(("train_loss", args.loss_threshold, args.consecutive))
    rows_b, rows_c = [], []
    for (metric, th, k) in criteria:
        cname = f"{metric}_{th}_k{k}"
        for arch in ARCHS:
            ar = [r for r in lab if r["config"]["arch"] == arch]
            steps0 = [convergence_step(r["log"], metric, th, k) for r in ar if r["config"]["p"] == 0.0]
            steps0 = [s for s in steps0 if s is not None]
            T0 = sum(steps0) / len(steps0) if steps0 else float("nan")   # mean over seeds at p=0
            for r in sorted(ar, key=lambda r: (r["config"]["p"], r["config"]["seed"])):
                c = r["config"]
                s = convergence_step(r["log"], metric, th, k)
                cens = s is None
                T = c["budget_steps"] if cens else s     # censored -> lower bound (budget), flagged
                rows_b.append(dict(criterion=cname, arch=arch, p=c["p"], seed=c["seed"],
                                   conv_step=s, censored=int(cens), T0_mean=T0,
                                   t_rel=T / T0, t_rel_is_lower_bound=int(cens)))
                if cname == f"{criteria[0][0]}_{criteria[0][1]}_k{criteria[0][2]}":
                    test_at = None
                    if s is not None:
                        test_at = next(rec["test_err"] for rec in r["log"] if rec["step"] == s)
                    rows_c.append(dict(arch=arch, p=c["p"], seed=c["seed"],
                                       final_test_err=r["final_test_err"], test_err_at_convergence=test_at,
                                       realized_changed_frac=r["realized_changed_frac"],
                                       diverged=int(r["diverged"])))
    write_csv(os.path.join(args.out, "fig1b.csv"), rows_b)
    # ---- Fig 1c: test error (= generalization error when train err = 0) ---
    write_csv(os.path.join(args.out, "fig1c.csv"), rows_c)

    # ---- Sanity summary against Table 1 anchors ---------------------------
    print("\nAnchor check (mean final test error; Table 1-implied p=0 reference):")
    for arch in ARCHS:
        for p in (0.0, 1.0):
            v = [x["final_test_err"] for x in rows_c if x["arch"] == arch and x["p"] == p]
            if v:
                ref = TABLE1_REF[arch] if p == 0.0 else 0.90
                print(f"  {arch} p={p}: ours {sum(v) / len(v):.4f} vs reference ~{ref:.4f}")
    n_cens = sum(r["censored"] for r in rows_b if r["criterion"] == f"{criteria[0][0]}_{criteria[0][1]}_k{criteria[0][2]}")
    print(f"censored runs under primary criterion: {n_cens}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    def common(sp):
        sp.add_argument("--out", default="results")
        sp.add_argument("--data-root", default="./data")
        sp.add_argument("--batch-size", type=int, default=128)
        sp.add_argument("--metric", default="train_acc", choices=["train_acc", "train_loss"])
        sp.add_argument("--threshold", type=float, default=0.99)
        sp.add_argument("--consecutive", type=int, default=2)
    sp = sub.add_parser("validate"); sp.add_argument("--data-root", default="./data")
    sp = sub.add_parser("sweep"); common(sp)
    sp.add_argument("--tier", default="full", choices=list(TIERS))
    sp.add_argument("--archs", nargs="*", choices=ARCHS)
    sp.add_argument("--p", nargs="*", type=float)
    sp.add_argument("--seeds", type=int)
    sp.add_argument("--budget", type=int, default=BUDGET_1B1C)
    sp = sub.add_parser("fig1a"); common(sp)
    sp.add_argument("--tier", default="full", choices=list(TIERS))
    sp.add_argument("--seeds", type=int)
    sp = sub.add_parser("aggregate"); common(sp)
    sp.add_argument("--loss-threshold", type=float, default=None)
    args = ap.parse_args()
    {"validate": cmd_validate, "sweep": cmd_sweep, "fig1a": cmd_fig1a,
     "aggregate": cmd_aggregate}[args.cmd](args)


if __name__ == "__main__":
    main()