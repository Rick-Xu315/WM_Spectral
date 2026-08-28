<h1 align="center">How do World Models and Policies Compose in LLM Agents?<br>A Joint Spectral and Behavioral Account</h1>

<p align="center">
  <!-- TODO: once the preprint is up, wrap this badge in
       <a href="https://arxiv.org/abs/ARXIV_ID"> ... </a> -->
  <img src="https://img.shields.io/badge/arXiv-coming%20soon-b31b1b.svg" alt="arXiv">
  <img src="https://img.shields.io/badge/EMNLP%202026-Findings-4b44ce.svg" alt="EMNLP 2026 Findings">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <img src="https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c.svg" alt="PyTorch">
</p>

Official code release for the EMNLP 2026 Findings paper.

This is an analysis-centric paper: the repository contains the **parameter-space
analysis and intervention code** (effective rank, subspace projection, rank
truncation, magnitude pruning, model merging). Training, data collection, and
rollout evaluation live in external frameworks and are not re-released here
(see [Model preparation](#model-preparation)).

## 📁 Repository layout

```
analysis/        Effective-rank analysis of post-training updates (RQ1)
interventions/   Spectral interventions that build modified checkpoints (RQ2)
merging/         KnOTS/TIES model merging (RQ3), forked from gstoica27/KnOTS
```

## ⚙️ Setup

```bash
pip install -r requirements.txt
```

<a id="model-preparation"></a>
## 🤖 Model preparation

All analyses operate on the checkpoint family of the two-stage
world-model / policy pipeline (base, WMRL, PRL, sequential WMRL->PRL, and the
WMRL setting sweep). These checkpoints are trained exactly as in
**Reinforcement World Model Learning for LLM-based Agents (RWML)**,
Yu et al., 2026 — [arXiv:2602.05842](https://arxiv.org/abs/2602.05842) — using
the [verl-agent](https://github.com/langfengq/verl-agent) framework for both
GRPO stages and for the ALFWorld ID/OOD rollout evaluation. Please refer to
that paper and repository for training data collection, hyperparameters, and
evaluation; the hyperparameters used here are also listed in the appendix of
our paper.

All scripts operate on HuggingFace-format checkpoints of
**Qwen2.5-7B-Instruct** (ALFWorld) or **Qwen3-8B** (tau^2-Bench). The examples
below assume this layout:

```
checkpoints/alfworld/{wmrl, wmsft, prl, wmsft_prl, wmrl_prl}
checkpoints/tau2/{wmrl, wmsft, prl, wmsft_prl, wmrl_prl}
```

How each entry point finds those checkpoints differs:

* `analysis/erank_perform.py` sweeps a whole checkpoint family, so its paths
  live in the `ALFWORLD_MODEL_SPECS` / `TAU2_MODEL_SPECS` dicts and are
  resolved against `--ckpt-root` (default `checkpoints/`). Only the `base`
  entry falls back to `from_pretrained` when the path is absent, so it may be
  a hub id; any other entry that is missing on disk is skipped with a warning
  (or raises under `--strict-paths`).
* `merging/configs/*.py` set their paths at the top of the config, resolved
  against the `WM_CKPT_ROOT` env var (default `<repo>/checkpoints`, an
  absolute path so it stays correct when you run from `merging/`). The base
  model comes from `WM_BASE_MODEL` (default `Qwen/Qwen2.5-7B-Instruct`).
* The `interventions/*.py` scripts take **explicit checkpoint paths** on the
  command line (`--base`, `--wmrl`, `--policy`, `--prl`, `--wprl`, `--anchor`,
  and `--tokenizer` for `magnitude_pruning.py`); they have no checkpoint-root
  option, only `--output-root` / `--ckpt-out` for where results are written.

Notation, following the paper: `Delta_X = theta_X - theta_0` (base model
`theta_0`); `DeltaDelta_PRL = (W+PRL) - WMRL` is the policy stage's conditional
update on top of the world model. All analyses operate module-wise on the seven
trainable linear projections `{q,k,v,o,up,gate,down}_proj`.

## 🗺️ Paper <-> code mapping

| Experiment | Code |
|---|---|
| Effective rank vs. task success across the WM-stage checkpoint sweep | `analysis/erank_perform.py` |
| Per-layer effective rank of each post-training paradigm's update | `analysis/erank_perform.py` (`--benchmark tau2` for the tau^2-Bench replication) |
| Left-space (output-direction) projection interventions | `interventions/subspace_projection.py --mode ckpts --side left` |
| Right-space (input-feature) projection interventions | `interventions/subspace_projection.py --mode ckpts --side right` |
| Module-wise subspace overlap analysis (principal angles, Frobenius cosine, mask IoU) | `interventions/subspace_projection.py --mode analysis` (`--compare prl-vs-ddprl` for the Delta_PRL vs. DeltaDelta_PRL pair) |
| Truncated rank-k pruning of the updates (WkP / WPk / WkPk / Pk) | `interventions/rank_truncation.py` |
| Magnitude pruning of the updates + sign-only baseline | `interventions/magnitude_pruning.py` |
| Training-free preservation merges: TIES / KnOTS(U) / KnOTS(V) | `merging/` (see [Model merging](#model-merging)) |

Training-time experiments (two-stage WMRL/PRL training, the online WM-SFT
auxiliary loss) and all end-to-end rollout evaluations are run with the
external RWML / verl-agent stack — see
[Model preparation](#model-preparation).

## 🚀 Running the experiments

**1. Effective rank.** Edit the `ALFWORLD_MODEL_SPECS` / `TAU2_MODEL_SPECS`
paths in `analysis/erank_perform.py` if your checkpoint layout differs (add
your WM-stage sweep variants there too), then:

```bash
python analysis/erank_perform.py --ckpt-root checkpoints --output effective_rank_alfworld.csv
python analysis/erank_perform.py --benchmark tau2 --ckpt-root checkpoints --output effective_rank_tau2.csv
```

The script outputs a single CSV (one row per model x layer x module); the
paper figures are simple aggregations/plots of this CSV.

**2. Subspace analysis + projection interventions.**

`--mode analysis` writes, into `--analysis-out`: per-module CSVs
(`principal_angles.csv`, `frobenius_cosine.csv`, `mask_iou.csv`), a
`summary.json` with the means, and five figures, each as both `.png` and
`.pdf` — `subspace_overlap_k{K}` (module-by-layer, one panel per side, at
`--overlap-k`), `frobenius_cosine`, `principal_angles_vs_k_{left,right}`, and
`mask_iou_heatmap`. `--mode ckpts` writes no analysis output; it only builds
checkpoints under `--ckpt-out`, named `{delta}_{side}_{projector}_k{k}`.

Note that `--side` applies only to `--mode ckpts`; the analysis phase always
reports both sides, so run it once rather than once per side.

```bash
# Geometry only (principal angles left/right, Frobenius cosine, mask IoU),
# here for the Delta_PRL vs. DeltaDelta_PRL overlap comparison
python interventions/subspace_projection.py --mode analysis \
    --base Qwen/Qwen2.5-7B-Instruct \
    --wmrl checkpoints/alfworld/wmrl \
    --policy checkpoints/alfworld/wmrl_prl --delta ddprl \
    --prl checkpoints/alfworld/prl --compare prl-vs-ddprl \
    --analysis-out subspace_result

# Projection checkpoints, e.g. left-space projections of DeltaDelta_PRL
python interventions/subspace_projection.py --mode ckpts --side left \
    --base Qwen/Qwen2.5-7B-Instruct \
    --wmrl checkpoints/alfworld/wmrl \
    --policy checkpoints/alfworld/wmrl_prl --delta ddprl \
    --ks 1,25,50,100,150,250,500 --ckpt-out checkpoints/projection

# Same for Delta_PRL: --policy checkpoints/alfworld/prl --delta prl
# Right-space versions: --side right
```

Each produced checkpoint is then evaluated end-to-end with the verl-agent
rollout pipeline (see [Model preparation](#model-preparation)).

**3. Rank truncation.**

```bash
python interventions/rank_truncation.py --variants WkP,WPk,WkPk,Pk --ks 1,25,50,250,500 \
    --base Qwen/Qwen2.5-7B-Instruct --wmrl checkpoints/alfworld/wmrl \
    --wprl checkpoints/alfworld/wmrl_prl --prl checkpoints/alfworld/prl \
    --output-root checkpoints/rank_truncate
```

**4. Magnitude pruning.**

```bash
python interventions/magnitude_pruning.py \
    --anchor checkpoints/alfworld/wmrl --policy checkpoints/alfworld/wmrl_prl \
    --tokenizer Qwen/Qwen2.5-7B-Instruct --output-root checkpoints/prune/delta_delta_prl
python interventions/magnitude_pruning.py \
    --anchor Qwen/Qwen2.5-7B-Instruct --policy checkpoints/alfworld/prl \
    --tokenizer Qwen/Qwen2.5-7B-Instruct --output-root checkpoints/prune/delta_prl
```

<a id="model-merging"></a>
## 🔀 Model merging

The training-free preservation experiments merge M_WMRL with a policy
checkpoint in weight space. `merging/` is a trimmed fork of
[KnOTS](https://github.com/gstoica27/KnOTS) ("Model merging with SVD to tie
the KnOTS", ICLR 2025; MIT license). Our changes on top of the original repo:

* **full-model merging** — a `FilteredFullModelHandler` and a Qwen causal-LM
  loader so fully fine-tuned checkpoints (rather than LoRA adapters or vision
  heads) can be merged, restricted to the seven transformer linear projections;
* **U and V merges** — alongside the original shared-U (output-side) KnOTS
  basis we add a shared-V (input-side) variant, motivated by the paper's
  geometric finding that the two stages share input (V) directions while
  writing to disjoint output (U) directions, selectable via `--shared-basis u|v`;
* **`merge_full_model.py`** — a CLI entry point that runs a two-checkpoint
  merge from a config and exports the merged model in HF format.

Only the three merge operations reported in the paper are kept. To reproduce
the WMRL + (WMRL->PRL) pairing (paths are set in the configs; `WM_CKPT_ROOT`
overrides the checkpoint root):

```bash
cd merging
# TIES
python merge_full_model.py --config-name qwen_wmrl_wprl_ties \
    --output-dir OUT/ties-wprl
# KnOTS (U)
python merge_full_model.py --config-name qwen_wmrl_wprl_ties \
    --representation svd-vector --shared-basis u --output-dir OUT/knots-u-wprl
# KnOTS (V)
python merge_full_model.py --config-name qwen_wmrl_wprl_ties \
    --representation svd-vector --shared-basis v --output-dir OUT/knots-v-wprl
```

Use `--config-name qwen_wmrl_prl_ties` for the (WMRL, PRL-from-base) pairing.
See `merging/README.md` for details.

* Projection/pruning checkpoints copy their non-target tensors (embeddings,
  lm_head, layernorms) from the policy checkpoint the delta is taken **from**,
  not from the anchor it is taken against. Each script's docstring spells out
  the pairing it uses.

## 📚 Citation

```bibtex
@inproceedings{xu2026wmspectral,
  title     = {How do World Models and Policies Compose in LLM Agents? A Joint Spectral and Behavioral Account},
  author    = {Xu, Ruize and Yu, Xiao and Tang, Yujin and Shang, Chenming and Singh, Nikhil},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2026},
  year      = {2026}
}
```
