"""Effective rank of post-training updates.

Paper experiments: (i) effective rank vs. task success across the WM-stage
checkpoint sweep, and (ii) per-layer / per-module effective rank of each
post-training paradigm's update — on ALFWorld (Qwen2.5-7B-Instruct) and, via
`--benchmark tau2`, on tau^2-Bench (Qwen3-8B).

For every checkpoint X we form the additive update Delta_X = theta_X - theta_0
module-wise on the trainable linear projections, and report
eRank(Delta) = exp(-sum_i p_i log p_i) with p_i the spectrum-normalized
singular-value mass (Roy & Vetterli, 2007).

Output: a single CSV with one row per (model, layer, module) and its
effective rank. Aggregation and plotting are left to downstream analysis.

Usage:
  python erank_perform.py --ckpt-root /path/to/checkpoints --output effective_rank.csv

Checkpoint paths in the *_MODEL_SPECS dicts are relative to --ckpt-root; edit
them to match your local layout.
"""

import argparse
import gc
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
import torch
from transformers import AutoModelForCausalLM


TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "up_proj",
    "gate_proj",
    "down_proj",
]

# Which submodule of a transformer block each projection lives on. Everything
# not listed here is looked up on the MLP, so TARGET_MODULES stays the single
# place to edit when changing the module set.
_ATTENTION_MODULES = frozenset({"q_proj", "k_proj", "v_proj", "o_proj"})


# ALFWorld / Qwen2.5-7B-Instruct checkpoint family (main benchmark).
# "base" may also point at the HF hub id if you have not downloaded it.
# For the WM-stage setting sweep, add your extra WMRL variants here
# (e.g. "WMRL(t_0.5)": "alfworld/wmrl_thresh0.5").
ALFWORLD_MODEL_SPECS = OrderedDict(
    {
        "base": "Qwen/Qwen2.5-7B-Instruct",
        "WMRL": "alfworld/wmrl",
        "WMSFT": "alfworld/wmsft",
        "PolicyRL": "alfworld/prl",
        "WMSFT+PolicyRL": "alfworld/wmsft_prl",
        "WMRL+PolicyRL": "alfworld/wmrl_prl",
    }
)

# tau^2-Bench / Qwen3-8B checkpoint family (replication benchmark).
TAU2_MODEL_SPECS = OrderedDict(
    {
        "base": "Qwen/Qwen3-8B",
        "PolicyRL": "tau2/prl",
        "WMSFT": "tau2/wmsft",
        "WMRL": "tau2/wmrl",
        "WMSFT+PolicyRL": "tau2/wmsft_prl",
        "WMRL+PolicyRL": "tau2/wmrl_prl",
    }
)

BENCHMARK_SPECS = {
    "alfworld": ALFWORLD_MODEL_SPECS,
    "tau2": TAU2_MODEL_SPECS,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute the effective rank of each checkpoint's update against the base model."
    )
    parser.add_argument("--benchmark", choices=list(BENCHMARK_SPECS), default="alfworld")
    parser.add_argument(
        "--ckpt-root",
        default="checkpoints",
        help="Root directory the relative paths in the *_MODEL_SPECS dicts resolve against.",
    )
    parser.add_argument(
        "--output",
        default="effective_rank.csv",
        help="Output CSV path (one row per model x layer x module).",
    )
    parser.add_argument(
        "--svd-device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device used for SVD (`cuda` or `cpu`).",
    )
    parser.add_argument(
        "--model-dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
        help="Checkpoint loading dtype. Updates are still cast to float32 before SVD.",
    )
    parser.add_argument(
        "--include-model",
        action="append",
        default=None,
        help="Optional model name filter. Can be passed multiple times.",
    )
    parser.add_argument(
        "--strict-paths",
        action="store_true",
        help="Fail instead of skipping missing checkpoints.",
    )
    return parser.parse_args()


def resolve_torch_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[dtype_name]


def effective_rank(singular_values: torch.Tensor) -> float:
    total = singular_values.sum()
    if total <= 0:
        return 0.0
    probs = singular_values / total
    entropy = -(probs * torch.log(probs + 1e-12)).sum()
    return torch.exp(entropy).item()


def compute_update_effective_rank(update: torch.Tensor, svd_device: str) -> float:
    singular_values = torch.linalg.svdvals(update.to(svd_device, dtype=torch.float32))
    erank = effective_rank(singular_values)
    del singular_values
    return erank


def load_model(path: str, dtype: torch.dtype) -> AutoModelForCausalLM:
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=dtype,
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()
    return model


def get_target_modules(model: AutoModelForCausalLM) -> Dict[int, Dict[str, torch.nn.Module]]:
    """{layer_id: {module_name: nn.Module}} for every name in TARGET_MODULES."""
    target = {}
    for layer_id, layer in enumerate(model.model.layers):
        modules = {}
        for name in TARGET_MODULES:
            parent = layer.self_attn if name in _ATTENTION_MODULES else layer.mlp
            if not hasattr(parent, name):
                raise AttributeError(
                    f"layer {layer_id} has no {name}; check TARGET_MODULES against "
                    f"this architecture."
                )
            modules[name] = getattr(parent, name)
        target[layer_id] = modules
    return target


def resolve_path(spec_path: str, ckpt_root: str) -> str:
    """Relative spec paths resolve against --ckpt-root; hub ids / absolute paths pass through."""
    p = Path(spec_path)
    if p.is_absolute() or p.exists():
        return str(p)
    local = Path(ckpt_root) / spec_path
    if local.exists():
        return str(local)
    # Fall back to the raw string (e.g. a HF hub id such as Qwen/Qwen2.5-7B-Instruct).
    return spec_path


def filter_model_specs(
    model_specs: OrderedDict,
    ckpt_root: str,
    include_models: Optional[Iterable[str]],
    strict_paths: bool,
) -> "OrderedDict[str, str]":
    filtered = OrderedDict()
    include_set = set(include_models) if include_models else None

    for model_name, spec_path in model_specs.items():
        if include_set is not None and model_name != "base" and model_name not in include_set:
            continue
        resolved = resolve_path(spec_path, ckpt_root)
        if not Path(resolved).exists():
            if model_name == "base":
                # The base entry may be a HF hub id; let from_pretrained resolve it.
                filtered[model_name] = resolved
                continue
            message = f"Checkpoint path does not exist for `{model_name}`: {resolved}"
            if strict_paths:
                raise FileNotFoundError(message)
            print(f"[WARN] {message}. Skipping.")
            continue
        filtered[model_name] = resolved

    if "base" not in filtered:
        raise RuntimeError("Base model is required for update effective rank analysis.")
    if len(filtered) <= 1:
        raise RuntimeError("No non-base models are available after filtering.")
    return filtered


@torch.inference_mode()
def analyze_effective_rank(
    model_specs: "OrderedDict[str, str]",
    model_dtype: torch.dtype,
    svd_device: str,
) -> pd.DataFrame:
    print(f"[INFO] Loading base model from {model_specs['base']}")
    base_model = load_model(model_specs["base"], model_dtype)
    base_modules = get_target_modules(base_model)

    records: List[Dict[str, object]] = []
    layer_ids = sorted(base_modules.keys())

    for model_name, path in model_specs.items():
        if model_name == "base":
            continue

        print(f"[INFO] Loading model `{model_name}` from {path}")
        model = load_model(path, model_dtype)
        model_modules = get_target_modules(model)

        for layer_id in layer_ids:
            for module_name in TARGET_MODULES:
                update = (
                    model_modules[layer_id][module_name].weight.detach().float()
                    - base_modules[layer_id][module_name].weight.detach().float()
                )
                records.append(
                    {
                        "model": model_name,
                        "layer": layer_id,
                        "module": module_name,
                        "effective_rank": compute_update_effective_rank(update, svd_device),
                    }
                )
                del update

        del model_modules
        del model
        gc.collect()
        if svd_device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    del base_modules
    del base_model
    gc.collect()
    if svd_device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()

    return pd.DataFrame.from_records(records)


def main() -> None:
    args = parse_args()

    model_specs = filter_model_specs(
        BENCHMARK_SPECS[args.benchmark],
        ckpt_root=args.ckpt_root,
        include_models=args.include_model,
        strict_paths=args.strict_paths,
    )

    details_df = analyze_effective_rank(
        model_specs=model_specs,
        model_dtype=resolve_torch_dtype(args.model_dtype),
        svd_device=args.svd_device,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    details_df.to_csv(output, index=False)
    print(f"[OK] Saved {len(details_df)} rows to {output}")


if __name__ == "__main__":
    main()
