"""Parameter handlers for fine-tuned models.

Derived from https://github.com/gstoica27/KnOTS (ft_handlers.py). Only our
addition is kept: ``FilteredFullModelHandler`` treats a fully fine-tuned
checkpoint as the task model but restricts merging to a configurable set of
target modules (embeddings, lm_head, layernorms, and biases stay untouched).
"""

from collections import OrderedDict

from torch import nn


class FilteredFullModelHandler(nn.Module):
    def __init__(self, base_model, target_modules=None, include_bias=False, output_device=None):
        super().__init__()
        self.base_model = base_model
        self.target_modules = tuple(target_modules or [])
        self.include_bias = include_bias
        self.output_device = output_device

    def _matches_target(self, key):
        if not self.target_modules:
            return True

        if not key.endswith(".weight"):
            if self.include_bias and key.endswith(".bias"):
                pass
            else:
                return False

        for module_name in self.target_modules:
            if f".{module_name}." in key or key.endswith(f".{module_name}.weight"):
                return True
            if self.include_bias and key.endswith(f".{module_name}.bias"):
                return True
        return False

    def get_ft_parameters(self):
        filtered = OrderedDict()
        for key, value in sorted(self.base_model.state_dict().items()):
            if self._matches_target(key):
                if self.output_device is not None:
                    value = value.to(self.output_device, non_blocking=True)
                filtered[key] = value
        return filtered

    def get_final_model(self, **kwargs):
        return self.base_model
