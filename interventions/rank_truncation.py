"""Truncated rank-k pruning of the post-training updates.

Paper experiment — how much of each update's task value survives low-rank
truncation:
  For each rank k, replace an update Delta by its Eckart-Young best rank-k
  approximation Delta_k (per target module) and assemble four checkpoint
  families for end-to-end evaluation:

    WkP  :  theta_0 + (Delta_WMRL)_k + DeltaDelta_PRL      (prune the WM update only)
    WPk  :  theta_0 + Delta_WMRL + (DeltaDelta_PRL)_k      (prune the policy update only)
    WkPk :  theta_0 + (Delta_WMRL)_k + (DeltaDelta_PRL)_k  (prune both)
    Pk   :  theta_0 + (Delta_PRL)_k                        (prune the from-base policy update)

  Non-target tensors are copied from W+PRL (WkP/WPk/WkPk) or PRL (Pk).

Example:
  python rank_truncation.py --variants WkP,WPk,WkPk,Pk --ks 1,25,50,250,500 \
      --base Qwen/Qwen2.5-7B-Instruct --wmrl CKPTS/alfworld/wmrl \
      --wprl CKPTS/alfworld/wmrl_prl --prl CKPTS/alfworld/prl \
      --output-root checkpoints/rank_truncate
"""

import argparse
import gc
import time
from pathlib import Path
from typing import Dict, List

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj",
                  "up_proj", "gate_proj", "down_proj")
SENTINEL_KEY = "model.layers.0.mlp.down_proj.weight"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--base", required=True, help="Base model path or HF hub id.")
    parser.add_argument("--wmrl", required=True, help="WMRL checkpoint.")
    parser.add_argument("--wprl", required=True, help="Sequential WMRL->PRL checkpoint.")
    parser.add_argument("--prl", default=None, help="PRL-from-base checkpoint (needed for Pk).")
    parser.add_argument("--variants", default="WkP,WPk,WkPk,Pk",
                        help="Comma-separated subset of {WkP,WPk,WkPk,Pk}.")
    parser.add_argument("--ks", default="1,25,50,250,500", help="Comma-separated truncation ranks.")
    parser.add_argument("--output-root", default="checkpoints/rank_truncate")
    parser.add_argument("--svd-device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def is_target_key(key: str) -> bool:
    if not key.endswith(".weight"):
        return False
    return any(f".{m}." in key or key.endswith(f".{m}.weight") for m in TARGET_MODULES)


def truncate_rank_k(delta_fp32: torch.Tensor, k: int, device: str) -> torch.Tensor:
    """Best rank-k approximation of delta (k is clamped to the matrix rank)."""
    d = delta_fp32.to(device, dtype=torch.float32)
    U, s, Vh = torch.linalg.svd(d, full_matrices=False)
    keep = min(k, s.numel())
    approx = (U[:, :keep] * s[:keep]) @ Vh[:keep]
    out = approx.to("cpu", dtype=torch.float32)
    del d, U, s, Vh, approx
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return out


def load_sd(path: str, target_keys: List[str] = None):
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16, device_map="cpu",
        low_cpu_mem_usage=True, trust_remote_code=True,
    )
    sd = model.state_dict()
    keys = target_keys if target_keys is not None else sorted(k for k in sd if is_target_key(k))
    return model, {k: sd[k].detach().clone() for k in keys}, keys


def main():
    args = parse_args()
    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    need_prl = any(v == "Pk" for v in variants)
    if need_prl and args.prl is None:
        raise ValueError("variant Pk requires --prl")

    print(f"[load] base <- {args.base}")
    base_model, base_sd, target_keys = load_sd(args.base)
    del base_model
    gc.collect()
    print(f"[load] WMRL <- {args.wmrl}")
    wmrl_model, wmrl_sd, _ = load_sd(args.wmrl, target_keys)
    del wmrl_model
    gc.collect()

    print(f"[load] W+PRL <- {args.wprl}")
    # Keep one full model in memory for save_pretrained; its pristine state
    # dict supplies the non-target tensors.
    carrier_model = AutoModelForCausalLM.from_pretrained(
        args.wprl, torch_dtype=torch.bfloat16, device_map="cpu",
        low_cpu_mem_usage=True, trust_remote_code=True,
    )
    wprl_sd_pristine = {k: v.detach().clone() for k, v in carrier_model.state_dict().items()}

    prl_sd_pristine: Dict[str, torch.Tensor] = {}
    if need_prl:
        print(f"[load] PRL <- {args.prl}")
        prl_model, _, _ = load_sd(args.prl, target_keys)
        prl_sd_pristine = {k: v.detach().clone() for k, v in prl_model.state_dict().items()}
        del prl_model
        gc.collect()

    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)

    grid = [(v, k) for v in variants for k in ks]
    for variant, k in tqdm(grid, desc="variants", unit="ckpt"):
        name = f"{variant}_k{k}"
        out_dir = out_root / name
        if (out_dir / "model.safetensors.index.json").exists():
            tqdm.write(f"[skip] {name}: already exists")
            continue

        t0 = time.time()
        donor = prl_sd_pristine if variant == "Pk" else wprl_sd_pristine
        new_sd = dict(donor)  # non-target tensors from the corresponding policy ckpt

        for key in tqdm(target_keys, desc=f"  {name}", leave=False, unit="mod"):
            base_v = base_sd[key].to(torch.float32)
            wmrl_v = wmrl_sd[key].to(torch.float32)
            if variant == "Pk":
                delta_prl = donor[key].to(torch.float32) - base_v
                new_val = base_v + truncate_rank_k(delta_prl, k, args.svd_device)
            else:
                delta_w = wmrl_v - base_v
                delta_dd = wprl_sd_pristine[key].to(torch.float32) - wmrl_v
                if variant in ("WkP", "WkPk"):
                    delta_w = truncate_rank_k(delta_w, k, args.svd_device)
                if variant in ("WPk", "WkPk"):
                    delta_dd = truncate_rank_k(delta_dd, k, args.svd_device)
                new_val = base_v + delta_w + delta_dd
            new_sd[key] = new_val.to(donor[key].dtype)
            del new_val

        out_dir.mkdir(parents=True, exist_ok=True)
        carrier_model.load_state_dict(new_sd, strict=True)
        carrier_model.save_pretrained(out_dir, safe_serialization=True)
        tokenizer.save_pretrained(out_dir)

        if SENTINEL_KEY in new_sd:
            b = base_sd[SENTINEL_KEY].to(torch.float32)
            d = donor[SENTINEL_KEY].to(torch.float32)
            n = new_sd[SENTINEL_KEY].to(torch.float32)
            tqdm.write(f"  [sentinel] ||new-base||_F={(n - b).norm():.4f} "
                       f"||new-donor||_F={(n - d).norm():.4f}")
        tqdm.write(f"[ok] {name} -> {out_dir} ({time.time() - t0:.1f}s)")
        del new_sd
        gc.collect()


if __name__ == "__main__":
    main()
