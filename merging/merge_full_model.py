import argparse
import os
from copy import deepcopy

import torch
from transformers import AutoTokenizer

from task_merger import get_merge_handler
from utils import get_config_from_name, prepare_models, prepare_param_handler


def parse_args():
    parser = argparse.ArgumentParser(description="Merge full-model checkpoints and export the merged model.")
    parser.add_argument("--config-name", required=True, help="Config module name under merging/configs.")
    parser.add_argument("--output-dir", required=True, help="Directory to save the merged model.")
    parser.add_argument("--device", default="cpu", help="Merge device for KnOTS internals.")
    parser.add_argument("--topk", type=float, default=None, help="Override task_merge_config.topK.")
    parser.add_argument("--scaling-coeff", type=float, default=None, help="Override shared scaling coefficient.")
    parser.add_argument(
        "--representation",
        choices=["vector", "svd-vector"],
        default=None,
        help="Override task_merge_config.representation.",
    )
    parser.add_argument(
        "--merge-method",
        choices=["tv", "ties"],
        default=None,
        help="Override task_merge_config.merge_method.",
    )
    parser.add_argument(
        "--shared-basis",
        choices=["u", "v"],
        default=None,
        help="KnOTS shared SVD basis: 'u' = output-side (original KnOTS), "
             "'v' = input-side variant. Sets task_merge_config.concat_across_output.",
    )
    parser.add_argument(
        "--ingredients-path",
        default=None,
        help="Optional path for cached SVD ingredients when using svd-vector.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    raw_config = get_config_from_name(args.config_name, device=args.device)
    config = deepcopy(raw_config)
    config["model"].setdefault("ft_config", {})
    config["model"]["ft_config"]["device"] = args.device

    if args.topk is not None:
        config["task_merge_config"]["topK"] = args.topk
    if args.scaling_coeff is not None:
        config["task_merge_config"]["scaling_coeffs"] = [args.scaling_coeff]
    if args.representation is not None:
        config["task_merge_config"]["representation"] = args.representation
    if args.merge_method is not None:
        config["task_merge_config"]["merge_method"] = args.merge_method
    if args.shared_basis is not None:
        config["task_merge_config"]["concat_across_output"] = (args.shared_basis == "u")
    if args.ingredients_path is not None:
        config["task_merge_config"]["ingredients_path"] = args.ingredients_path

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[main] loading models for config={args.config_name}")
    models = prepare_models(config["model"], device=args.device)
    print("[main] models loaded")
    MergeClass = get_merge_handler(config["task_merge_config"]["representation"])
    Merge = MergeClass(
        [model.cpu().eval() for model in models["bases"]],
        pretrained_model=models["new"].cpu().eval(),
        param_handler=prepare_param_handler(config["model"].get("ft_config", {})),
        device=args.device,
        merge_config=config["task_merge_config"],
    )

    Merge.set_scaling_coeffs(config["task_merge_config"]["scaling_coeffs"])
    if config["task_merge_config"]["representation"] == "svd-vector":
        print("[main] running transform() for svd-vector merge")
        Merge.transform(config["task_merge_config"])

    with torch.no_grad():
        print("[main] running merge()")
        merged_model = Merge.merge(config["task_merge_config"])

    print(f"[main] saving merged model to {args.output_dir}")
    merged_model.save_pretrained(args.output_dir, safe_serialization=True)
    print("[main] saving tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"].get("ptm_path") or config["model"]["name"],
        trust_remote_code=config["model"].get("trust_remote_code", True),
    )
    tokenizer.save_pretrained(args.output_dir)

    print(f"[main] merged model written to: {args.output_dir}")


if __name__ == "__main__":
    main()
