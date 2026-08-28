"""Magnitude pruning / lottery-ticket experiments on the policy-stage updates.

Paper experiment — coordinate-space (magnitude-based) sparsification of the
policy-stage updates, complementing the spectral analyses:
  For Delta in {DeltaDelta_PRL, Delta_PRL}, retain a fraction p of the entries
  under three selection rules (top-p by |Delta|, bottom-p, random-p), plus a
  sign-only variant with per-module Frobenius-norm calibration
  (alpha = ||Delta||_F / sqrt(N)). Each variant is written out as a full
  checkpoint for end-to-end evaluation:

    anchor=WMRL, policy=W+PRL :  theta = WMRL + pruned(W+PRL - WMRL)
                                 (non-target tensors from W+PRL)
    anchor=base, policy=PRL   :  theta = base + pruned(PRL - base)
                                 (non-target tensors from PRL)

  5 top-p + 5 bottom-p + 5 random-p + 1 sign-only = 16 checkpoints per pairing.

Example:
  python magnitude_pruning.py \
      --anchor CKPTS/alfworld/wmrl --policy CKPTS/alfworld/wmrl_prl \
      --tokenizer Qwen/Qwen2.5-7B-Instruct --output-root checkpoints/prune/delta_delta_prl
"""

import argparse
import gc
import time
from pathlib import Path
from typing import Optional

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj",
                  "up_proj", "gate_proj", "down_proj")
P_VALUES = (0.5, 0.3, 0.1, 0.05, 0.02)
RANDOM_SEED = 0
SENTINEL_KEY = "model.layers.0.mlp.down_proj.weight"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--anchor", required=True,
                        help="Anchor checkpoint the pruned delta is re-added to "
                             "(WMRL to prune W+PRL - WMRL, the base model to prune PRL - base).")
    parser.add_argument("--policy", required=True,
                        help="Policy checkpoint the delta is taken from "
                             "(W+PRL against a WMRL anchor, PRL against a base anchor).")
    parser.add_argument("--tokenizer", required=True,
                        help="Tokenizer source (usually the base model path or hub id).")
    parser.add_argument("--output-root", default="checkpoints/prune")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def is_target_key(key: str) -> bool:
    if not key.endswith(".weight"):
        return False
    return any(f".{m}." in key or key.endswith(f".{m}.weight") for m in TARGET_MODULES)


def top_p_pruned_delta(delta_fp32: torch.Tensor, p: float) -> torch.Tensor:
    abs_d = delta_fp32.abs()
    n = abs_d.numel()
    k = max(1, int(round(p * n)))
    threshold = torch.kthvalue(abs_d.view(-1), n - k + 1).values
    return delta_fp32 * (abs_d >= threshold)


def bottom_p_pruned_delta(delta_fp32: torch.Tensor, p: float) -> torch.Tensor:
    abs_d = delta_fp32.abs()
    n = abs_d.numel()
    k = max(1, int(round(p * n)))
    threshold = torch.kthvalue(abs_d.view(-1), k).values
    return delta_fp32 * (abs_d <= threshold)


def random_p_pruned_delta(delta_fp32: torch.Tensor, p: float,
                          generator: torch.Generator) -> torch.Tensor:
    n = delta_fp32.numel()
    k = max(1, int(round(p * n)))
    perm = torch.randperm(n, generator=generator, device=delta_fp32.device)
    mask = torch.zeros(n, dtype=torch.bool, device=delta_fp32.device)
    mask[perm[:k]] = True
    return delta_fp32 * mask.view_as(delta_fp32)


def sign_only_delta(delta_fp32: torch.Tensor) -> torch.Tensor:
    # alpha calibrated so ||alpha * sign(Delta)||_F == ||Delta||_F per module.
    nonzero = delta_fp32 != 0
    n_nonzero = int(nonzero.sum().item())
    if n_nonzero == 0:
        return torch.zeros_like(delta_fp32)
    alpha = (delta_fp32[nonzero].norm() / (n_nonzero ** 0.5)).item()
    return alpha * torch.sign(delta_fp32)


def transform_for_variant(variant_name: str, delta_fp32: torch.Tensor,
                          generator: Optional[torch.Generator]) -> torch.Tensor:
    if variant_name.startswith("topp_"):
        return top_p_pruned_delta(delta_fp32, float(variant_name.split("_")[1]))
    if variant_name.startswith("bottomp_"):
        return bottom_p_pruned_delta(delta_fp32, float(variant_name.split("_")[1]))
    if variant_name.startswith("randomp_"):
        assert generator is not None
        return random_p_pruned_delta(delta_fp32, float(variant_name.split("_")[1]), generator)
    if variant_name == "sign_only":
        return sign_only_delta(delta_fp32)
    raise ValueError(f"unknown variant {variant_name}")


def variant_list():
    """5 top-p + 5 bottom-p + 5 random-p + 1 sign-only = 16 variants."""
    out = []
    for p in P_VALUES:
        out.append((f"topp_{p}", None))
        out.append((f"bottomp_{p}", None))
        out.append((f"randomp_{p}", RANDOM_SEED + int(p * 10000)))
    out.append(("sign_only", None))
    return out


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    device = args.device
    print(f"[device] computing on: {device}")

    print(f"[load] anchor <- {args.anchor}")
    anchor_model = AutoModelForCausalLM.from_pretrained(
        args.anchor, torch_dtype=torch.bfloat16,
        device_map="cpu", low_cpu_mem_usage=True, trust_remote_code=True,
    )
    anchor_full = anchor_model.state_dict()
    target_keys = [k for k in anchor_full if is_target_key(k)]
    # Anchor target tensors live on the compute device — reused across all 16
    # variants, so the one-time transfer cost is amortized.
    anchor_target_dev = {k: anchor_full[k].detach().to(device) for k in target_keys}
    del anchor_full, anchor_model
    gc.collect()
    print(f"[load] anchor target tensors on {device}: {len(target_keys)}")

    print(f"[load] policy <- {args.policy}")
    policy_model = AutoModelForCausalLM.from_pretrained(
        args.policy, torch_dtype=torch.bfloat16,
        device_map="cpu", low_cpu_mem_usage=True, trust_remote_code=True,
    )
    # Pristine policy snapshot supplies the non-target tensors of each ckpt.
    policy_sd_pristine = {k: v.detach().clone()
                          for k, v in policy_model.state_dict().items()}
    policy_target_dev = {k: policy_sd_pristine[k].to(device) for k in target_keys}

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    for variant_name, generator_seed in tqdm(variant_list(), desc="variants", unit="ckpt"):
        out_dir = output_root / variant_name
        if (out_dir / "model.safetensors.index.json").exists():
            tqdm.write(f"[skip] {variant_name}: already exists at {out_dir}")
            continue

        t0 = time.time()
        generator = None
        if generator_seed is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(generator_seed)

        new_sd = dict(policy_sd_pristine)
        for key in tqdm(target_keys, desc=f"  {variant_name}", leave=False, unit="mod"):
            anchor_dev = anchor_target_dev[key]
            policy_dev = policy_target_dev[key]
            delta = policy_dev.to(torch.float32) - anchor_dev.to(torch.float32)
            new_delta = transform_for_variant(variant_name, delta, generator)
            new_val = (anchor_dev.to(torch.float32) + new_delta).to(policy_dev.dtype)
            new_sd[key] = new_val.detach().to("cpu")
            del delta, new_delta, new_val

        if device.startswith("cuda"):
            torch.cuda.empty_cache()

        out_dir.mkdir(parents=True, exist_ok=True)
        policy_model.load_state_dict(new_sd)
        policy_model.save_pretrained(out_dir, safe_serialization=True)
        tokenizer.save_pretrained(out_dir)

        if SENTINEL_KEY in new_sd:
            a = anchor_target_dev[SENTINEL_KEY].to(torch.float32)
            p = policy_target_dev[SENTINEL_KEY].to(torch.float32)
            n = new_sd[SENTINEL_KEY].to(device, dtype=torch.float32)
            tqdm.write(f"  [sentinel] ||new-anchor||_F={(n - a).norm():.4f} "
                       f"||new-policy||_F={(n - p).norm():.4f}")
        tqdm.write(f"[ok] {variant_name} -> {out_dir} ({time.time() - t0:.1f}s)")
        del new_sd
        gc.collect()


if __name__ == "__main__":
    main()
