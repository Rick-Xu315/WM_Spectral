"""KnOTS-TIES merge of (WMRL, sequential WMRL->PRL) on Qwen2.5-7B-Instruct.

Paper experiment: training-free world-model preservation merges over (M_WMRL, M_WMRL+PRL).
  * representation "vector" + merge_method "ties"          -> plain TIES
  * --representation svd-vector --shared-basis u           -> KnOTS(U)
  * --representation svd-vector --shared-basis v           -> KnOTS(V)
  ("concat_across_output" below is the U/V default; the CLI flag overrides it.)

Edit WM_CKPT_ROOT (env var) or the paths below to point at your checkpoints.
"""

import os
from pathlib import Path

# Default to <repo>/checkpoints so the path stays correct when these
# merges are launched from the merging/ directory.
CKPT_ROOT = os.environ.get(
    "WM_CKPT_ROOT", str(Path(__file__).resolve().parents[2] / "checkpoints"))

PTM_PATH = os.environ.get("WM_BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
WMRL_PATH = f"{CKPT_ROOT}/alfworld/wmrl"
WMRL_POLICY_RL_PATH = f"{CKPT_ROOT}/alfworld/wmrl_prl"

config = {
    "model": {
        "name": "Qwen/Qwen2.5-7B-Instruct",
        "arch": "qwen2_causallm",
        "ptm_path": PTM_PATH,
        "cachedir": "",
        "torch_dtype": "bfloat16",
        "trust_remote_code": True,
        "bases": [
            WMRL_PATH,
            WMRL_POLICY_RL_PATH,
        ],
        "ft_config": {
            "type": "filtered_full",
            "target_modules": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "up_proj",
                "gate_proj",
                "down_proj",
            ],
            "include_bias": False,
        },
    },
    "task_merge_config": {
        "representation": "vector",
        "sign_resolve_mode": "sum_of_values",
        "topK": 40,
        "merge_method": "ties",
        "merging_type": "mean",
        "scaling_coeffs": [1.0],
        "concat_across_output": True,
        "dare": False,
        "dare_pruning_coeffs": 0.0,
    },
}
