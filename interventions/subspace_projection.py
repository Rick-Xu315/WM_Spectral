"""Subspace geometry of Delta_WMRL vs. the policy-stage updates.

Paper experiments — the weight-space relationship between the world-model
update and the policy-stage updates:
  * Phase A ("--mode analysis") — module-wise principal-angle overlap between
    the leading-k left/right singular subspaces of two updates, the per-module
    Frobenius cosine between them, and top-p binary mask IoU. Produces the
    paper's module-by-layer overlap heatmaps (one panel per side, at
    --overlap-k), the Frobenius-cosine heatmap, and the aggregated alignment
    curves.
  * Phase B ("--mode ckpts") — the left/right projection-intervention
    checkpoints. For a projection rank k, the policy-stage update is replaced
    by its parallel / orthogonal / random-subspace projection with respect to
    Delta_WMRL's leading left (output) or right (input-feature) singular
    directions, and the recombined model is saved for end-to-end evaluation:

      --delta ddprl :  theta_0 + Delta_WMRL + proj(DeltaDelta_PRL)
                       (DeltaDelta_PRL = W+PRL - WMRL; non-target tensors from W+PRL)
      --delta prl   :  theta_0 + proj(Delta_PRL)
                       (Delta_PRL = PRL - base; non-target tensors from PRL)

    Left side:  parallel = U_k U_k^T @ Delta,  orthogonal = Delta - parallel
    Right side: parallel = Delta @ V_k V_k^T,  orthogonal = Delta - parallel
    Random:     U_k / V_k replaced by a random orthonormal k-frame.

Example:
  python subspace_projection.py --mode both --side left \
      --base Qwen/Qwen2.5-7B-Instruct --wmrl CKPTS/alfworld/wmrl \
      --policy CKPTS/alfworld/wmrl_prl --delta ddprl \
      --ks 1,25,50,100,150,250,500 \
      --analysis-out subspace_result --ckpt-out checkpoints/projection
"""

import argparse
import gc
import json
import time
import zlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402


TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj",
                  "up_proj", "gate_proj", "down_proj")
K_ANALYSIS = (1, 5, 10, 25, 50, 100, 150, 250, 500)
P_IOU = (0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5)
RANDOM_SEED_BASE = 0
SENTINEL_KEY = "model.layers.0.mlp.down_proj.weight"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--mode", choices=["analysis", "ckpts", "both"], default="both")
    parser.add_argument("--base", required=True, help="Base model path or HF hub id.")
    parser.add_argument("--wmrl", required=True, help="WMRL checkpoint (defines the projector subspace).")
    parser.add_argument("--policy", required=True,
                        help="Policy-stage checkpoint: W+PRL for --delta ddprl, PRL for --delta prl.")
    parser.add_argument("--delta", choices=["ddprl", "prl"], default="ddprl",
                        help="Which policy update to project (see module docstring).")
    parser.add_argument("--prl", default=None,
                        help="Optional PRL-from-base checkpoint, needed for --compare prl-vs-ddprl.")
    parser.add_argument("--compare", choices=["wmrl-vs-policy", "prl-vs-ddprl"],
                        default="wmrl-vs-policy",
                        help="Delta pair used in analysis mode: Delta_WMRL vs the policy update "
                             "(default), or Delta_PRL vs DeltaDelta_PRL (the module-wise overlap "
                             "comparison between the two policy updates; requires --prl and "
                             "--delta ddprl).")
    parser.add_argument("--side", choices=["left", "right"], default="left",
                        help="Which side to project onto in --mode ckpts: Delta_WMRL's left "
                             "(output) or right (input-feature) singular subspace. Ignored by "
                             "--mode analysis, which always reports both sides.")
    parser.add_argument("--ks", default="1,25,50,100,150,250,500",
                        help="Comma-separated projection ranks for checkpoint construction.")
    parser.add_argument("--projectors", default="parallel,orthogonal,random",
                        help="Comma-separated subset of {parallel,orthogonal,random}.")
    parser.add_argument("--overlap-k", type=int, default=1, choices=K_ANALYSIS,
                        help="Rank used for the module-by-layer overlap heatmaps. Must be one "
                             f"of the analysed ranks {list(K_ANALYSIS)}; default 1.")
    parser.add_argument("--analysis-out", default="subspace_result")
    parser.add_argument("--ckpt-out", default="checkpoints/projection")
    parser.add_argument("--svd-device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def is_target_key(key: str) -> bool:
    if not key.endswith(".weight"):
        return False
    return any(f".{m}." in key or key.endswith(f".{m}.weight") for m in TARGET_MODULES)


def parse_layer_module(key: str) -> Tuple[Optional[int], Optional[str]]:
    """`model.layers.<i>.{self_attn,mlp}.<module>.weight` -> (i, module)."""
    parts = key.split(".")
    try:
        i = parts.index("layers")
        return int(parts[i + 1]), parts[i + 3]
    except (ValueError, IndexError):
        return None, None


def svd_bases(delta_fp32: torch.Tensor, device: str,
              max_k: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """fp32 SVD on `device`; return top-max_k left (U) and right (V) singular vectors on CPU.

    The third element is the *requested* max_k. A module whose rank is below
    max_k yields fewer columns than requested, so callers must compare against
    this value (not the column count) to tell a stale cache entry from one that
    is simply rank-limited.
    """
    d = delta_fp32.to(device, dtype=torch.float32)
    U, s, Vh = torch.linalg.svd(d, full_matrices=False)
    keep = min(max_k, int(U.shape[1]))
    U_top = U[:, :keep].detach().to("cpu", dtype=torch.float32).contiguous()
    V_top = Vh[:keep].T.detach().to("cpu", dtype=torch.float32).contiguous()
    del U, s, Vh, d
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return U_top, V_top, max_k


def principal_angle_stats(B_a: torch.Tensor, B_b: torch.Tensor, k: int,
                          device: str) -> Tuple[float, float]:
    """mean(sigma^2) and min(sigma) of the principal cosines between two k-frames."""
    Ba = B_a[:, :k].to(device, dtype=torch.float32)
    Bb = B_b[:, :k].to(device, dtype=torch.float32)
    sigma = torch.linalg.svdvals(Ba.T @ Bb).detach().to("cpu")
    del Ba, Bb
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return float((sigma ** 2).mean().item()), float(sigma.min().item())


def top_p_bool_mask(abs_d_flat: torch.Tensor, p: float) -> torch.Tensor:
    n = abs_d_flat.numel()
    k = max(1, int(round(p * n)))
    threshold = torch.kthvalue(abs_d_flat, n - k + 1).values
    return abs_d_flat >= threshold


def mask_iou(abs_a: torch.Tensor, abs_b: torch.Tensor, p: float, device: str) -> float:
    a, b = abs_a.to(device), abs_b.to(device)
    m1, m2 = top_p_bool_mask(a, p), top_p_bool_mask(b, p)
    inter = (m1 & m2).sum().item()
    union = (m1 | m2).sum().item()
    del a, b, m1, m2
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return float(inter) / float(union) if union > 0 else 0.0


def frobenius_cosine(delta_a: torch.Tensor, delta_b: torch.Tensor) -> float:
    """<A,B>_F / (||A||_F ||B||_F) for two updates at the same module."""
    a, b = delta_a.reshape(-1), delta_b.reshape(-1)
    denom = float(a.norm()) * float(b.norm())
    return float(torch.dot(a, b)) / denom if denom > 0 else 0.0


def _module_layer_heatmap(ax, df: pd.DataFrame, value_col: str, title: str,
                          vmin: float, vmax: float, cmap: str):
    """Modules on the y axis, layers on the x axis (the paper's heatmap layout)."""
    pivot = df.pivot_table(index="module", columns="layer", values=value_col)
    pivot = pivot.reindex([m for m in TARGET_MODULES if m in pivot.index])
    im = ax.imshow(pivot.values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([m.replace("_proj", "") for m in pivot.index], fontsize=9)
    ax.set_xticks(range(0, len(pivot.columns), max(1, len(pivot.columns) // 8)))
    ax.set_xticklabels([str(pivot.columns[i])
                        for i in range(0, len(pivot.columns),
                                       max(1, len(pivot.columns) // 8))])
    ax.set_title(title, fontsize=10)
    return im


def plot_overlap_panels(df_pa: pd.DataFrame, k: int, out_stem: Path) -> None:
    """Module x layer rank-k subspace overlap, left (output) above right (input)."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)
    im = None
    for ax, side in zip(axes, ("left", "right")):
        sub = df_pa[(df_pa["side"] == side) & (df_pa["k"] == k)]
        if sub.empty:
            continue
        im = _module_layer_heatmap(
            ax, sub, "mean_sigma_sq",
            f"{side.capitalize()} rank-{k} subspace overlap "
            f"(mean {sub['mean_sigma_sq'].mean():.2f})",
            0.0, 1.0, "viridis")
    axes[-1].set_xlabel("Layer")
    if im is not None:
        fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    fig.savefig(out_stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_frobenius_cosine(df_fc: pd.DataFrame, out_stem: Path) -> None:
    """Module x layer per-module Frobenius cosine between the two updates."""
    lim = max(0.05, float(df_fc["frobenius_cosine"].abs().max()))
    fig, ax = plt.subplots(figsize=(10, 3.2))
    im = _module_layer_heatmap(
        ax, df_fc, "frobenius_cosine",
        f"Per-module Frobenius cosine (mean {df_fc['frobenius_cosine'].mean():.3f})",
        -lim, lim, "coolwarm")
    ax.set_xlabel("Layer")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="Frob. cos.")
    fig.tight_layout()
    fig.savefig(out_stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(df: pd.DataFrame, value_col: str, x_col: str,
                 out_stem: Path, title: str) -> None:
    """Diagnostic sweep view: one row per (layer, module), one column per k or p."""
    pivot = df.pivot_table(index=["layer", "module"], columns=x_col, values=value_col)
    pivot = pivot.sort_index(level=["layer", "module"])
    fig, ax = plt.subplots(figsize=(8, max(8, 0.05 * len(pivot.index) + 4)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(c) for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"L{l}/{m}" for l, m in pivot.index], fontsize=4)
    fig.colorbar(im, ax=ax)
    ax.set_xlabel(x_col)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_aggregated(df_pa: pd.DataFrame, df_iou: pd.DataFrame, out_stem: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(19, 5))
    for ax, side in zip(axes[:2], ("left", "right")):
        sub = df_pa[df_pa["side"] == side]
        agg = sub.groupby("k")["mean_sigma_sq"].agg(["mean", "min", "max"]).sort_index()
        ax.plot(agg.index, agg["mean"], marker="o", linewidth=2, label="mean over modules")
        ax.fill_between(agg.index, agg["min"], agg["max"], alpha=0.25, label="min/max over modules")
        ax.set_xlabel("k (top singular vectors)")
        ax.set_ylabel("Subspace alignment mean(sigma^2)")
        ax.set_title(f"{side.capitalize()}-side principal-angle alignment vs k")
        ax.grid(alpha=0.3)
        ax.legend()

    agg_i = df_iou.groupby("p")["iou"].agg(["mean", "min", "max"]).sort_index()
    axes[2].plot(agg_i.index, agg_i["mean"], marker="o", linewidth=2, label="mean over modules")
    axes[2].fill_between(agg_i.index, agg_i["min"], agg_i["max"], alpha=0.25, label="min/max over modules")
    axes[2].set_xlabel("p (top fraction by magnitude)")
    axes[2].set_ylabel("Binary mask IoU")
    axes[2].set_title("Top-p mask IoU vs p")
    axes[2].grid(alpha=0.3)
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(out_stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def phase_analysis(target_keys: List[str],
                   delta_a_fn, delta_b_fn,
                   out_root: Path, svd_device: str,
                   basis_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor, int]],
                   overlap_k: int = 1) -> None:
    """Principal-angle overlap (left & right), Frobenius cosine, and mask IoU."""
    out_root.mkdir(parents=True, exist_ok=True)
    pa_records, iou_records, fc_records = [], [], []
    max_k = max(K_ANALYSIS)

    pbar = tqdm(target_keys, desc="[analysis] SVD + IoU", unit="mod")
    for idx, key in enumerate(pbar):
        layer_idx, module_name = parse_layer_module(key)
        if layer_idx is None:
            continue
        pbar.set_postfix(layer=layer_idx, mod=module_name)

        delta_a = delta_a_fn(key)
        delta_b = delta_b_fn(key)

        if key not in basis_cache or basis_cache[key][2] < max_k:
            basis_cache[key] = svd_bases(delta_a, svd_device, max_k=max_k)
        UA, VA, _ = basis_cache[key]
        UB, VB, _ = svd_bases(delta_b, svd_device, max_k=max_k)

        for k in K_ANALYSIS:
            for side, (Ba, Bb) in (("left", (UA, UB)), ("right", (VA, VB))):
                k_eff = min(k, int(Ba.shape[1]), int(Bb.shape[1]))
                if k_eff < 1:
                    continue
                mean_s2, min_s = principal_angle_stats(Ba, Bb, k_eff, svd_device)
                pa_records.append({
                    "layer": layer_idx, "module": module_name, "side": side,
                    "k": k, "k_effective": k_eff,
                    "mean_sigma_sq": mean_s2, "min_sigma": min_s,
                })

        fc_records.append({
            "layer": layer_idx, "module": module_name,
            "frobenius_cosine": frobenius_cosine(delta_a, delta_b),
        })

        abs_a = delta_a.abs().view(-1)
        abs_b = delta_b.abs().view(-1)
        for p in P_IOU:
            iou_records.append({
                "layer": layer_idx, "module": module_name, "p": p,
                "iou": mask_iou(abs_a, abs_b, p, svd_device),
            })

        del UB, VB, delta_a, delta_b, abs_a, abs_b
        if (idx + 1) % 14 == 0:
            gc.collect()
    pbar.close()

    df_pa = pd.DataFrame(pa_records)
    df_iou = pd.DataFrame(iou_records)
    df_fc = pd.DataFrame(fc_records)
    df_pa.to_csv(out_root / "principal_angles.csv", index=False)
    df_iou.to_csv(out_root / "mask_iou.csv", index=False)
    df_fc.to_csv(out_root / "frobenius_cosine.csv", index=False)

    # Paper-layout figures: module x layer, one panel per side.
    plot_overlap_panels(df_pa, overlap_k, out_root / f"subspace_overlap_k{overlap_k}")
    plot_frobenius_cosine(df_fc, out_root / "frobenius_cosine")
    # Diagnostic sweeps over k / p.
    for side in ("left", "right"):
        plot_heatmap(df_pa[df_pa["side"] == side], "mean_sigma_sq", "k",
                     out_root / f"principal_angles_vs_k_{side}",
                     title=f"{side}-side mean(sigma^2) per module")
    plot_heatmap(df_iou, "iou", "p", out_root / "mask_iou_heatmap",
                 title="top-p mask IoU per module")
    plot_aggregated(df_pa, df_iou, out_root / "aggregated_alignment")

    summary = {
        "principal_angles_by_side_k": {
            f"{side}/k={k}": round(
                float(df_pa[(df_pa["side"] == side) & (df_pa["k"] == k)]["mean_sigma_sq"].mean()), 6)
            for side in ("left", "right") for k in K_ANALYSIS
        },
        "frobenius_cosine_mean": round(float(df_fc["frobenius_cosine"].mean()), 6),
        "iou_by_p": {str(p): round(float(df_iou[df_iou["p"] == p]["iou"].mean()), 6)
                     for p in P_IOU},
        "n_modules": len(target_keys),
    }
    with (out_root / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"[analysis] done -> {out_root}")


def available_rank(delta_fp32: torch.Tensor, basis: Optional[torch.Tensor],
                   projector_type: str, side: str) -> int:
    """Largest projector rank this module can actually support on `side`."""
    dim = int(delta_fp32.shape[0]) if side == "left" else int(delta_fp32.shape[1])
    if projector_type == "random":
        return dim
    assert basis is not None
    return int(basis.shape[1])


def project_delta(delta_fp32: torch.Tensor,
                  basis: Optional[torch.Tensor], projector_type: str,
                  side: str, k: int, key: str, device: str) -> torch.Tensor:
    """Apply the rank-k parallel/orthogonal/random projector on the chosen side.

    k is clamped to the rank the module can actually support (`available_rank`);
    phase_ckpts reports up front how many modules a given k affects.
    """
    dim = int(delta_fp32.shape[0]) if side == "left" else int(delta_fp32.shape[1])
    k_eff = min(k, available_rank(delta_fp32, basis, projector_type, side))
    if projector_type == "random":
        # Clamp before drawing the frame: qr(mode="reduced") on a (dim, k>dim)
        # matrix returns only dim columns, which would silently turn the
        # "rank-k" random control into a full-rank (identity) projector.
        # The seed keys off the requested k so variants stay reproducible.
        seed = RANDOM_SEED_BASE + k * 100003 + (zlib.crc32(key.encode()) & 0xFFFF)
        g = torch.Generator(device="cpu").manual_seed(seed)
        R = torch.randn(dim, k_eff, generator=g, dtype=torch.float32)
        Q, _ = torch.linalg.qr(R, mode="reduced")
        B = Q.to(device, dtype=torch.float32)
    else:
        assert basis is not None
        B = basis[:, :k_eff].to(device, dtype=torch.float32)

    delta_dev = delta_fp32.to(device, dtype=torch.float32)
    if side == "left":
        parallel = B @ (B.T @ delta_dev)
    else:
        parallel = (delta_dev @ B) @ B.T

    new_delta = delta_dev - parallel if projector_type == "orthogonal" else parallel
    out = new_delta.to("cpu", dtype=torch.float32)
    del B, delta_dev, parallel, new_delta
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return out


def phase_ckpts(args, target_keys, anchor_sd, policy_sd_pristine,
                policy_model, tokenizer,
                basis_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor, int]],
                delta_wmrl_fn) -> None:
    """Build theta_anchor + proj(policy delta) checkpoints for each (projector, k)."""
    ckpt_root = Path(args.ckpt_out)
    ckpt_root.mkdir(parents=True, exist_ok=True)
    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    projectors = [x.strip() for x in args.projectors.split(",") if x.strip()]
    max_k = max(ks)

    # Ensure the Delta_WMRL SVD basis covers every target key at the required rank.
    # Compare against the rank the cached entry was *requested* at, not its column
    # count: a module whose rank is below max_k can never satisfy the latter and
    # would be re-decomposed on every run.
    need = [key for key in target_keys
            if key not in basis_cache or basis_cache[key][2] < max_k]
    for idx, key in enumerate(tqdm(need, desc="[ckpts] precompute Delta_WMRL SVD", unit="mod")):
        basis_cache[key] = svd_bases(delta_wmrl_fn(key), args.svd_device,
                                     max_k=max(max_k, max(K_ANALYSIS)))
        if (idx + 1) % 14 == 0:
            gc.collect()

    grid = [(t, k) for t in projectors for k in ks]
    for projector_type, k in tqdm(grid, desc="[ckpts] variants", unit="ckpt"):
        variant = f"{args.delta}_{args.side}_{projector_type}_k{k}"
        out_dir = ckpt_root / variant
        if (out_dir / "model.safetensors.index.json").exists():
            tqdm.write(f"[skip] {variant}: already exists")
            continue

        # Surface the rank clamp up front rather than projecting onto fewer
        # than k directions without saying so.
        ranks = [available_rank(policy_sd_pristine[key], basis_cache[key][0],
                                projector_type, args.side) for key in target_keys]
        clamped = [r for r in ranks if r < k]
        if clamped:
            tqdm.write(
                f"[warn] {variant}: k={k} exceeds the available rank for "
                f"{len(clamped)}/{len(target_keys)} modules (min {min(clamped)}); "
                f"those modules use their full rank."
            )

        t0 = time.time()
        new_sd = dict(policy_sd_pristine)  # non-target tensors from the policy ckpt
        for key in tqdm(target_keys, desc=f"  {variant}", leave=False, unit="mod"):
            anchor_val = anchor_sd[key]
            policy_val = policy_sd_pristine[key]
            delta = policy_val.to(torch.float32) - anchor_val.to(torch.float32)
            basis = None
            if projector_type != "random":
                U, V, _ = basis_cache[key]
                basis = U if args.side == "left" else V
            new_delta = project_delta(delta, basis, projector_type,
                                      args.side, k, key, args.svd_device)
            new_sd[key] = (anchor_val.to(torch.float32) + new_delta).to(policy_val.dtype)
            del delta, new_delta

        out_dir.mkdir(parents=True, exist_ok=True)
        policy_model.load_state_dict(new_sd)
        policy_model.save_pretrained(out_dir, safe_serialization=True)
        tokenizer.save_pretrained(out_dir)

        if SENTINEL_KEY in new_sd:
            a = anchor_sd[SENTINEL_KEY].to(torch.float32)
            p = policy_sd_pristine[SENTINEL_KEY].to(torch.float32)
            n = new_sd[SENTINEL_KEY].to(torch.float32)
            tqdm.write(f"  [sentinel] ||new-anchor||_F={(n - a).norm():.4f} "
                       f"||new-policy||_F={(n - p).norm():.4f}")
        tqdm.write(f"[ok] {variant} -> {out_dir} ({time.time() - t0:.1f}s)")
        del new_sd
        gc.collect()


def load_target_sd(path: str, target_keys: Optional[List[str]] = None):
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16, device_map="cpu",
        low_cpu_mem_usage=True, trust_remote_code=True,
    )
    sd = model.state_dict()
    keys = target_keys if target_keys is not None else sorted(k for k in sd if is_target_key(k))
    return model, {k: sd[k].detach().clone() for k in keys}, keys


def main():
    args = parse_args()

    print(f"[load] base <- {args.base}")
    base_model, base_sd, target_keys = load_target_sd(args.base)
    del base_model
    gc.collect()

    print(f"[load] WMRL <- {args.wmrl}")
    wmrl_model, wmrl_sd, _ = load_target_sd(args.wmrl, target_keys)
    del wmrl_model
    gc.collect()

    print(f"[load] policy ({args.delta}) <- {args.policy}")
    policy_model = AutoModelForCausalLM.from_pretrained(
        args.policy, torch_dtype=torch.bfloat16, device_map="cpu",
        low_cpu_mem_usage=True, trust_remote_code=True,
    )
    policy_sd_pristine = {k: v.detach().clone() for k, v in policy_model.state_dict().items()}
    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)

    def delta_wmrl(key):
        return wmrl_sd[key].to(torch.float32) - base_sd[key].to(torch.float32)

    # The projected update and the anchor it is re-added to (see docstring).
    if args.delta == "ddprl":
        anchor_sd = wmrl_sd

        def delta_policy(key):
            return policy_sd_pristine[key].to(torch.float32) - wmrl_sd[key].to(torch.float32)
    else:
        anchor_sd = base_sd

        def delta_policy(key):
            return policy_sd_pristine[key].to(torch.float32) - base_sd[key].to(torch.float32)

    basis_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor, int]] = {}

    if args.mode in ("analysis", "both"):
        if args.compare == "prl-vs-ddprl":
            # Module-wise overlap between Delta_PRL (= PRL - base) and
            # DeltaDelta_PRL (= W+PRL - WMRL). Requires --delta ddprl so that
            # delta_policy is DeltaDelta_PRL, plus the standalone PRL ckpt.
            if args.prl is None or args.delta != "ddprl":
                raise ValueError("--compare prl-vs-ddprl requires --prl and --delta ddprl")
            print(f"[load] PRL <- {args.prl}")
            prl_model, prl_sd, _ = load_target_sd(args.prl, target_keys)
            del prl_model
            gc.collect()

            def delta_prl(key):
                return prl_sd[key].to(torch.float32) - base_sd[key].to(torch.float32)

            # Separate cache: the shared basis_cache must stay Delta_WMRL-only
            # because phase_ckpts reuses it for the projector.
            phase_analysis(target_keys, delta_prl, delta_policy,
                           Path(args.analysis_out), args.svd_device, {},
                           overlap_k=args.overlap_k)
        else:
            phase_analysis(target_keys, delta_wmrl, delta_policy,
                           Path(args.analysis_out), args.svd_device, basis_cache,
                           overlap_k=args.overlap_k)

    if args.mode in ("ckpts", "both"):
        phase_ckpts(args, target_keys, anchor_sd, policy_sd_pristine,
                    policy_model, tokenizer, basis_cache, delta_wmrl)

    print("[done] subspace projection experiments complete")


if __name__ == "__main__":
    main()
