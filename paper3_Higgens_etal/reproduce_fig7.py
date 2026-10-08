#!/usr/bin/env python3
"""
Reproduction of Figure 7A (beta-VAE, beta=4) and Figure 7B (VAE, beta=1) from
"beta-VAE: Learning Basic Visual Concepts with a Constrained Variational
Framework" (Higgins et al., ICLR 2017), on the dSprites / "2D shapes" data.

FROM THE PAPER (Table 1, Sec. 4.2, Fig. 7 caption, App. A.1):
  * Data      : 737,280 binary 64x64 images (3 shapes x 6 scales x 40 rotations
                x 32 pos-X x 32 pos-Y) -> the official dSprites file.
  * Encoder   : 4096 -> FC 1200 -> FC 1200 (ReLU) -> 10 latents (Gaussian).
  * Decoder   : 10 -> FC 1200 -> FC 1200 -> FC 1200 (Tanh) -> FC 4096, Bernoulli.
  * Optimiser : Adagrad, learning rate 1e-2.
  * Objective : E_q[log p(x|z)] - beta * KL(q(z|x) || N(0, I)), beta = 4 vs 1.
  * Fig. 7    : columns = latents ordered by learnt Gaussian variance (shown in
                the last row); row 1 = mean of mu_m over the 32x32 positions
                (averaged over shapes/rotations/scales); row 2 = mu_m vs scale
                (averaged over rotations and positions); row 3 = mu_m vs rotation
                (averaged over scales and positions); square=red, oval=green,
                heart=blue; rows 4-8 = decoded traversal of one latent over
                [-3, 3] with the other 9 latents fixed to the values obtained by
                running inference on a dataset image.

ASSUMPTIONS (not specified in the paper; see README/summary in the chat):
  A1 batch size 64, 150k steps (--batch_size/--steps); A2 PyTorch default init;
  A3 decoder output layer is linear -> sigmoid (Bernoulli logits); A4 encoder
  outputs (mu, log sigma^2); A5 reconstruction term SUMMED over the 4096 pixels,
  KL summed over latents, both averaged over the batch (beta=4 is unnormalised;
  normalised beta = 4*10/4096 ~ 0.01); A6 inputs are raw {0,1} pixels;
  A7 "learnt variance" = mean over the whole dataset of the posterior sigma^2_m;
  A8 traversal: 5 evenly spaced values in [3, -3], other latents = posterior
  MEAN of the seed image; A9 seed image = a heart at the centre (configurable);
  A10 5 lowest-variance latents + the highest-variance latent as "Z_unused";
  A11 heat-map / curve axis ranges are chosen per panel from the data;
  A12 the factor labels under each column are computed automatically.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

IMG_PIXELS = 64 * 64
SHAPE_COLORS = ["red", "green", "blue"]  # square, oval(ellipse), heart (dSprites order 1,2,3)
SHAPE_NAMES = ["square", "oval", "heart"]


# --------------------------------------------------------------------------- #
# Data (numpy only). Images are bit-packed so the whole dataset is ~377 MB.
# --------------------------------------------------------------------------- #
def load_dsprites(npz_path, cache_dir):
    """Returns packed uint8 array (N, 512) and the dataset's latent sizes."""
    os.makedirs(cache_dir, exist_ok=True)
    c_img = os.path.join(cache_dir, "dsprites_packed.npy")
    c_meta = os.path.join(cache_dir, "dsprites_meta.npz")
    if os.path.exists(c_img) and os.path.exists(c_meta):
        print(f"[data] loading cache {c_img}")
        return np.load(c_img), np.load(c_meta)["sizes"]

    if not os.path.exists(npz_path):
        raise FileNotFoundError(
            f"{npz_path} not found. Download the official file "
            "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz from the "
            "google-deepmind/dsprites-dataset repository and pass --data_path.")
    print(f"[data] reading {npz_path} (decompresses ~3 GB once; later runs use the cache)")
    d = np.load(npz_path, allow_pickle=True, encoding="latin1")
    meta = d["metadata"][()]
    sizes = np.asarray(meta["latents_sizes"], dtype=np.int64)  # [color,shape,scale,orient,x,y]

    # Verify that image index == mixed-radix index of the latent classes, because
    # all aggregation below relies on reshape(shape, scale, rot, x, y).
    classes = d["latents_classes"].astype(np.int64)
    bases = np.concatenate([np.cumprod(sizes[::-1])[::-1][1:], [1]])
    assert np.array_equal(classes @ bases, np.arange(len(classes))), \
        "dataset is not in canonical Cartesian-product order"

    imgs = d["imgs"]
    n = imgs.shape[0]
    packed = np.empty((n, IMG_PIXELS // 8), dtype=np.uint8)
    for s in range(0, n, 50000):
        chunk = (imgs[s:s + 50000].reshape(-1, IMG_PIXELS) > 0)
        packed[s:s + 50000] = np.packbits(chunk, axis=1)
    del imgs
    np.save(c_img, packed)
    np.savez(c_meta, sizes=sizes)
    return packed, sizes


def unpack(packed_batch):
    """uint8 (B,512) tensor -> float32 (B,4096) of {0,1} (np.packbits is big-endian)."""
    shifts = torch.arange(7, -1, -1, device=packed_batch.device, dtype=torch.int32)
    bits = (packed_batch.to(torch.int32).unsqueeze(-1) >> shifts) & 1
    return bits.reshape(packed_batch.shape[0], -1).float()


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class BetaVAE(nn.Module):
    def __init__(self, zdim=10, hidden=1200):
        super().__init__()
        self.zdim = zdim
        self.enc = nn.Sequential(
            nn.Linear(IMG_PIXELS, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU())
        self.enc_out = nn.Linear(hidden, 2 * zdim)           # mu, log sigma^2   (A4)
        self.dec = nn.Sequential(
            nn.Linear(zdim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, IMG_PIXELS))                   # Bernoulli logits (A3)

    def encode(self, x):
        mu, logvar = self.enc_out(self.enc(x)).chunk(2, dim=1)
        return mu, logvar

    def decode(self, z):
        return self.dec(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return self.decode(z), mu, logvar


def beta_vae_loss(x, logits, mu, logvar, beta):
    b = x.shape[0]
    recon = F.binary_cross_entropy_with_logits(logits, x, reduction="sum") / b
    kl_per = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())   # (B, zdim)
    kl = kl_per.sum() / b
    return recon + beta * kl, recon, kl, kl_per.mean(0)


def train(beta, packed, args, device, ckpt_path):
    torch.manual_seed(args.seed)                 # same init for every beta
    model = BetaVAE(args.zdim).to(device)
    opt = torch.optim.Adagrad(model.parameters(), lr=args.lr)   # paper: Adagrad 1e-2
    step = 0
    if os.path.exists(ckpt_path) and not args.retrain:
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step = ck["step"]
        print(f"[beta={beta}] resumed from step {step}")
    if args.plot_only:
        if step == 0:
            raise RuntimeError(f"--plot_only but no checkpoint at {ckpt_path}")
        return model.eval()

    n = packed.shape[0]
    gen = torch.Generator().manual_seed(args.seed + step)
    perm, ptr = None, 0
    run = np.zeros(3)
    run_kl = torch.zeros(args.zdim, device=device)
    t0 = time.time()
    model.train()
    while step < args.steps:
        if perm is None or ptr + args.batch_size > n:
            perm, ptr = torch.randperm(n, generator=gen).to(device), 0
        idx = perm[ptr:ptr + args.batch_size]
        ptr += args.batch_size
        x = unpack(packed[idx])
        logits, mu, logvar = model(x)
        loss, recon, kl, kl_per = beta_vae_loss(x, logits, mu, logvar, beta)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        step += 1
        run += np.array([loss.item(), recon.item(), kl.item()]) if step % 50 == 0 else 0
        run_kl += kl_per.detach() if step % 50 == 0 else 0
        if step % args.log_every == 0:
            k = args.log_every / 50
            print(f"[beta={beta}] step {step}/{args.steps}  loss {run[0]/k:.2f}  "
                  f"recon {run[1]/k:.2f}  KL {run[2]/k:.2f}  "
                  f"({(time.time()-t0)/60:.1f} min)")
            print("      KL/latent:", np.round((run_kl / k).cpu().numpy(), 2))
            run[:] = 0
            run_kl.zero_()
        if step % args.ckpt_every == 0 or step == args.steps:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step}, ckpt_path)
    return model.eval()


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #
@torch.no_grad()
def encode_all(model, packed, device, bs=4096):
    mus, vs = [], []
    for s in range(0, packed.shape[0], bs):
        mu, logvar = model.encode(unpack(packed[s:s + bs]))
        mus.append(mu.cpu()); vs.append(logvar.exp().cpu())
    return torch.cat(mus).numpy(), torch.cat(vs).numpy()


@torch.no_grad()
def traversal_images(model, packed, seed_idx, latents, device, n_steps=5, rng=3.0):
    x = unpack(packed[seed_idx:seed_idx + 1])
    mu, _ = model.encode(x)                                   # A8: fixed latents = posterior mean
    vals = torch.linspace(rng, -rng, n_steps, device=device)  # top row = +3, bottom = -3
    out = np.zeros((len(latents), n_steps, 64, 64), dtype=np.float32)
    for j, m in enumerate(latents):
        z = mu.repeat(n_steps, 1)
        z[:, m] = vals
        out[j] = torch.sigmoid(model.decode(z)).reshape(n_steps, 64, 64).cpu().numpy()
    return out


def factor_scores(M):
    """M: (shape, scale, rot, x, y, L). Fraction of variance of mu_m explained by each factor
    (within-shape conditional-mean variance; for 'shape' the between-shape variance)."""
    L = M.shape[-1]
    tot = M.reshape(-1, L).var(axis=0) + 1e-12
    sc = {"shape": M.mean(axis=(1, 2, 3, 4)).var(axis=0) / tot}
    for name, ax in [("scale", 1), ("rotation", 2), ("pos. X", 3), ("pos. Y", 4)]:
        others = tuple(a for a in range(5) if a not in (0, ax))
        sc[name] = M.mean(axis=others).var(axis=1).mean(axis=0) / tot
    return sc, tot


def analyse(mu, var, sizes, n_show):
    S, Sc, R, X, Y = [int(v) for v in sizes[1:]]
    L = mu.shape[1]
    M = mu.astype(np.float64).reshape(S, Sc, R, X, Y, L)
    learnt_var = var.mean(0)                                           # A7
    kl = (0.5 * (mu ** 2 + var - 1 - np.log(var))).mean(0)
    order = np.argsort(learnt_var)
    cols = list(order[:n_show]) + [int(order[-1])]                     # A10
    scores, tot = factor_scores(M)
    names = list(scores.keys())
    labels = []
    for m in range(L):
        s = np.array([scores[k][m] for k in names])
        if tot[m] < 0.05:
            labels.append("unused")
        elif s.max() < 0.3:
            labels.append("mixed")
        else:
            labels.append(names[int(s.argmax())])
    return dict(
        cols=[int(c) for c in cols], learnt_var=learnt_var, kl=kl, labels=labels,
        scores={k: v.tolist() for k, v in scores.items()},
        pos_map=M.mean(axis=(0, 1, 2)),          # (X, Y, L)
        scale_curves=M.mean(axis=(2, 3, 4)),     # (S, Sc, L)
        rot_curves=M.mean(axis=(1, 3, 4)),       # (S, R, L)
    )


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def draw_panel(fig, res, trav, letter, title):
    cols = res["cols"]
    nC = len(cols)
    gs = fig.add_gridspec(9, nC, height_ratios=[1] * 8 + [0.6], hspace=0.14, wspace=0.12,
                          left=0.16, right=0.98, top=0.90, bottom=0.05)
    shown = cols
    heat = max(0.5, float(np.abs(res["pos_map"][:, :, shown]).max()))          # A11
    lim = max(0.5, float(max(np.abs(res["scale_curves"][:, :, shown]).max(),
                             np.abs(res["rot_curves"][:, :, shown]).max())))
    axs = [[fig.add_subplot(gs[r, c]) for c in range(nC)] for r in range(9)]
    for c, m in enumerate(cols):
        unused = (c == nC - 1)
        # row 1: position heat-map (rows = pos Y, columns = pos X), jet: red = high
        ax = axs[0][c]
        ax.imshow(res["pos_map"][:, :, m].T, cmap="jet", vmin=-heat, vmax=heat, interpolation="nearest")
        ax.set_title(r"$Z_{unused}$" if unused else rf"$z_{{{m + 1}}}$", fontsize=9)
        # row 2/3: scale and rotation curves, one line per shape
        for r, key in [(1, "scale_curves"), (2, "rot_curves")]:
            ax = axs[r][c]
            for s in range(res[key].shape[0]):
                ax.plot(res[key][s, :, m], color=SHAPE_COLORS[s], lw=0.8)
            ax.axhline(0, color="0.8", lw=0.4)
            ax.set_ylim(-lim, lim)
            ax.set_xlim(0, res[key].shape[1] - 1)
        # rows 4-8: traversal
        for r in range(5):
            ax = axs[3 + r][c]
            ax.imshow(trav[c][r], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        # last row: learnt variance + factor label
        ax = axs[8][c]
        ax.axis("off")
        ax.text(0.5, 0.75, f"{res['learnt_var'][m]:.2f}", ha="center", va="center", fontsize=7,
                transform=ax.transAxes)
        ax.text(0.5, 0.25, res["labels"][m], ha="center", va="center", fontsize=6,
                transform=ax.transAxes)
    for r in range(8):
        for c in range(nC):
            axs[r][c].set_xticks([]); axs[r][c].set_yticks([])
    for r, t in enumerate(["position", "scale", "rotation"]):
        axs[r][0].set_ylabel(t, fontsize=7)
    for r, t in [(3, "3"), (5, "0"), (7, "-3")]:
        axs[r][0].set_yticks([32]); axs[r][0].set_yticklabels([t], fontsize=7)
    axs[8][0].text(-0.15, 0.5, "learnt\nvariance", ha="right", va="center", fontsize=6,
                   transform=axs[8][0].transAxes)
    # group labels
    p0, p2 = axs[0][0].get_position(), axs[2][0].get_position()
    p3, p7 = axs[3][0].get_position(), axs[7][0].get_position()
    fig.text(0.035, (p0.y1 + p2.y0) / 2, "mean latent response", rotation=90, ha="center",
             va="center", fontsize=7, fontweight="bold")
    fig.text(0.035, (p3.y1 + p7.y0) / 2, "single latent mean traversal", rotation=90, ha="center",
             va="center", fontsize=7, fontweight="bold")
    fig.text(0.01, 0.975, letter, fontsize=13, fontweight="bold", va="top")
    fig.text(0.57, 0.975, title, fontsize=10, ha="center", va="top")
    fig.text(0.57, 0.012, "rows 2-3: square = red, oval = green, heart = blue",
             fontsize=6, ha="center", va="bottom")


def make_fig(res, trav, letter, title, path):
    fig = plt.figure(figsize=(5.2, 6.4))
    draw_panel(fig, res, trav, letter, title)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def make_combined(items, path):
    fig = plt.figure(figsize=(10.4, 6.4))
    subs = fig.subfigures(1, len(items))
    subs = np.atleast_1d(subs)
    for sf, (res, trav, letter, title) in zip(subs, items):
        draw_panel(sf, res, trav, letter, title)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", default="dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz")
    p.add_argument("--cache_dir", default="cache")
    p.add_argument("--out_dir", default="fig7_outputs")
    p.add_argument("--betas", type=float, nargs="+", default=[4.0, 1.0])
    p.add_argument("--steps", type=int, default=150_000)          # A1 (not in paper)
    p.add_argument("--batch_size", type=int, default=64)          # A1 (not in paper)
    p.add_argument("--lr", type=float, default=1e-2)              # paper: Adagrad 1e-2
    p.add_argument("--zdim", type=int, default=10)                # paper: 10 latents
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_show", type=int, default=5)               # informative columns shown
    p.add_argument("--seed_img", type=int, nargs=5, default=[2, 3, 0, 16, 16],
                   metavar=("SHAPE", "SCALE", "ROT", "X", "Y"),
                   help="indices of the traversal seed image (shape 0=square,1=oval,2=heart)")
    p.add_argument("--log_every", type=int, default=2000)
    p.add_argument("--ckpt_every", type=int, default=10_000)
    p.add_argument("--device", default=None)
    p.add_argument("--plot_only", action="store_true", help="skip training; load checkpoints")
    p.add_argument("--retrain", action="store_true", help="ignore existing checkpoints")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else
                          "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
                          else "cpu"))
    print("device:", device)
    packed_np, sizes = load_dsprites(args.data_path, args.cache_dir)
    packed = torch.from_numpy(packed_np).to(device)
    seed_flat = int(np.ravel_multi_index(tuple(args.seed_img), tuple(int(v) for v in sizes[1:])))

    panels, summary = {}, {}
    for beta in args.betas:
        tag = f"beta{beta:g}"
        model = train(beta, packed, args, device, os.path.join(args.out_dir, f"ckpt_{tag}.pt"))
        mu, var = encode_all(model, packed, device)
        res = analyse(mu, var, sizes, args.n_show)
        trav = [traversal_images(model, packed, seed_flat, [m], device)[0] for m in res["cols"]]
        panels[beta] = (res, trav)
        summary[tag] = {
            "latent_order_by_variance(1-indexed)": [c + 1 for c in res["cols"]],
            "learnt_variance_all": np.round(res["learnt_var"], 4).tolist(),
            "kl_nats_all": np.round(res["kl"], 4).tolist(),
            "auto_labels_all": res["labels"],
            "variance_explained_by_factor": {k: np.round(v, 3).tolist() for k, v in res["scores"].items()},
        }
        print(f"[{tag}] learnt variance:", np.round(res["learnt_var"], 2))
        print(f"[{tag}] labels:", res["labels"])

    letters = {4.0: ("A", r"$\beta$-VAE ($\beta$=4)"), 1.0: ("B", r"VAE ($\beta$=1)")}
    items = []
    for beta, (res, trav) in panels.items():
        letter, title = letters.get(beta, (f"b{beta:g}", rf"$\beta$-VAE ($\beta$={beta:g})"))
        name = f"figure7{letter}.png" if beta in letters else f"figure7_beta{beta:g}.png"
        make_fig(res, trav, letter, title, os.path.join(args.out_dir, name))
        items.append((res, trav, letter, title))
    if len(items) > 1:
        make_combined(items, os.path.join(args.out_dir, "figure7AB_combined.png"))
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
