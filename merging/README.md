# Model merging (KnOTS fork)

Trimmed fork of [KnOTS](https://github.com/gstoica27/KnOTS)
("Model merging with SVD to tie the KnOTS", ICLR 2025; MIT license, see
`LICENSE`), used for the paper's training-free world-model preservation
experiments. Only the three operations reported in the paper are kept:
**plain TIES**, **KnOTS(U)**, and **KnOTS(V)**.

## Our changes vs. the original repo

* **Full-model merging** — the original repo merges LoRA adapters /
  vision heads. We add `ft_handlers.FilteredFullModelHandler` (treats a fully
  fine-tuned checkpoint as the task model, restricted to the seven transformer
  linear projections `{q,k,v,o,up,gate,down}_proj`; embeddings, lm_head,
  layernorms and biases stay at their base values) and a Qwen causal-LM loader
  (`arch: qwen2_causallm`).
* **U- and V-sharing merges** — original KnOTS concatenates task deltas along
  the input dimension, giving a shared left/output basis U. We add a V-sharing
  variant (transpose before the joint SVD, so the shared basis lies on the
  input side), motivated by the paper's geometric finding that the two training
  stages share input (V) directions while writing to disjoint output (U)
  directions. Selected via `--shared-basis u|v` (`concat_across_output` in the
  config).
* **`merge_full_model.py`** — CLI entry point that runs a two-checkpoint merge
  from a config and exports the merged model in HF format.
* Everything unrelated (vision/NLI datasets, training and evaluation scripts,
  LoRA handlers) has been removed; see the original repo for those.

## Commands (WMRL + WMRL->PRL pairing)

Checkpoint paths are set in `configs/` (`WM_CKPT_ROOT` overrides the root).
`configs/qwen_wmrl_wprl_ties.py` merges (WMRL, WMRL->PRL);
`configs/qwen_wmrl_prl_ties.py` merges (WMRL, PRL-from-base).

```bash
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

All three use TIES top-K=40, `sum_of_values` sign resolution, disjoint-mean
aggregation, scaling coefficient 1.0, and no DARE (config defaults).
