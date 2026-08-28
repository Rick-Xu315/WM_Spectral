"""Slimmed-down KnOTS utilities.

Derived from https://github.com/gstoica27/KnOTS (utils.py). Only the pieces
needed for the full-model TIES / KnOTS merges in this release are kept:
config loading, Qwen causal-LM preparation, the parameter-handler factory,
and the merging/masking function registries.
"""

from copy import deepcopy
from inspect import getmembers, isfunction

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM


def get_config_from_name(name, device=None):
    """Load config based on its name (module under configs/)."""
    out = deepcopy(getattr(__import__('configs.' + name), name).config)
    if device is None and 'device' not in out:
        out['device'] = 'cuda'
    elif device is not None:
        out['device'] = device
    return out


def prepare_param_handler(ft_config):
    """Load FT model parameter extractors."""
    if ft_config.get('type', None) == 'filtered_full':
        from ft_handlers import FilteredFullModelHandler
        target_modules = ft_config.get('target_modules')
        include_bias = ft_config.get('include_bias', False)
        output_device = ft_config.get('device')
        return lambda model: FilteredFullModelHandler(
            model,
            target_modules=target_modules,
            include_bias=include_bias,
            output_device=output_device,
        )
    raise ValueError(
        f"Unsupported ft_config type: {ft_config.get('type')!r}. This release "
        "only ships the 'filtered_full' handler; see the original KnOTS repo "
        "for LoRA/FFT/vision handlers."
    )


def prepare_qwen_causallm(config, device):
    """Load Qwen causal LM full checkpoints from local paths."""
    del device
    bases = []
    dtype_name = config.get('torch_dtype', 'bfloat16')
    model_kwargs = {
        'torch_dtype': getattr(torch, dtype_name),
        'device_map': 'cpu',
        'low_cpu_mem_usage': True,
        'trust_remote_code': config.get('trust_remote_code', True),
    }

    ptm_path = config.get('ptm_path') or config['name']
    new_model = AutoModelForCausalLM.from_pretrained(ptm_path, **model_kwargs)

    for base_path in tqdm(config['bases'], desc="Preparing Qwen Models", position=0, leave=True):
        base_model = AutoModelForCausalLM.from_pretrained(base_path, **model_kwargs)
        base_model.eval()
        bases += [base_model]

    return {
        'bases': bases,
        'new': new_model.eval(),
    }


def prepare_models(config, device='cuda'):
    if config.get('arch') == 'qwen2_causallm':
        return prepare_qwen_causallm(config, device)
    raise ValueError(
        f"Unsupported model arch: {config.get('arch')!r}. This release only "
        "ships the Qwen causal-LM loader; see the original KnOTS repo for "
        "CLIP/ViT/Llama loaders."
    )


def get_merging_fn(name):
    """Get the merging function from name tag."""
    import merging_functions
    vector_fns = dict([(k.replace('_merging', ''), v)
                       for (k, v) in getmembers(merging_functions, isfunction)
                       if '_merging' in k])
    return vector_fns[name]


def get_mask_fn(name):
    """Get the masking function from name tag."""
    import masking_ops
    masking_fns = dict([(k.replace('_masking', ''), v)
                        for (k, v) in getmembers(masking_ops, isfunction)
                        if '_masking' in k])
    return masking_fns[name]
