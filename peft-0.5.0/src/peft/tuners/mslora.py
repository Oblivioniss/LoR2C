# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
import re
import warnings
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from typing import List, Optional, Tuple, Union
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers.pytorch_utils import Conv1D

from ..config import PeftConfig
from ..import_utils import is_bnb_4bit_available, is_bnb_available
from ..utils import (
    COMMON_LAYERS_PATTERN,
    TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
    TRANSFORMERS_MODELS_TO_MSLORA_TARGET_MODULES_MAPPING,
    ModulesToSaveWrapper,
    PeftType,
    _freeze_adapter,
    _get_submodules,
    get_auto_gptq_quant_linear,
    get_quantization_config,
    transpose,
)
from .tuners_utils import BaseTuner, BaseTunerLayer
from transformers.utils import logging
import time

if is_bnb_available():
    import bitsandbytes as bnb


@dataclass
class MSLoraConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`LoraModel`].

    Args:
        r (`int`): Lora attention dimension.
        target_modules (`Union[List[str],str]`): The names of the modules to apply Lora to.
        lora_alpha (`int`): The alpha parameter for Lora scaling.
        lora_dropout (`float`): The dropout probability for Lora layers.
        fan_in_fan_out (`bool`): Set this to True if the layer to replace stores weight like (fan_in, fan_out).
            For example, gpt-2 uses `Conv1D` which stores weights like (fan_in, fan_out) and hence this should be set
            to `True`.
        bias (`str`): Bias type for Lora. Can be 'none', 'all' or 'lora_only'. If 'all' or 'lora_only', the
            corresponding biases will be updated during training. Be aware that this means that, even when disabling
            the adapters, the model will not produce the same output as the base model would have without adaptation.
        modules_to_save (`List[str]`):List of modules apart from LoRA layers to be set as trainable
            and saved in the final checkpoint.
        layers_to_transform (`Union[List[int],int]`):
            The layer indexes to transform, if this argument is specified, it will apply the LoRA transformations on
            the layer indexes that are specified in this list. If a single integer is passed, it will apply the LoRA
            transformations on the layer at this index.
        layers_pattern (`str`):
            The layer pattern name, used only if `layers_to_transform` is different from `None` and if the layer
            pattern is not in the common layers pattern.
    """

    r: int = field(default=8, metadata={"help": "Lora attention dimension"})
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with Lora."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )
    lora_alpha: int = field(default=8, metadata={"help": "Lora alpha"})
    lora_dropout: float = field(default=0.0, metadata={"help": "Lora dropout"})
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"},
    )
    bias: str = field(default="none", metadata={"help": "Bias type for Lora. Can be 'none', 'all' or 'lora_only'"})
    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint. "
            "For example, in Sequence Classification or Token Classification tasks, "
            "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
        },
    )
    init_lora_weights: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to initialize the weights of the Lora layers with their default initialization. Don't change "
                "this setting, except if you know exactly what you're doing."
            ),
        },
    )
    layers_to_transform: Optional[Union[List[int], int]] = field(
        default=None,
        metadata={
            "help": "The layer indexes to transform, is this argument is specified, PEFT will transform only the layers indexes that are specified inside this list. If a single integer is passed, PEFT will transform only the layer at this index."
        },
    )
    layers_pattern: Optional[str] = field(
        default=None,
        metadata={
            "help": "The layer pattern name, used only if `layers_to_transform` is different to None and if the layer pattern is not in the common layers pattern."
        },
    )

    def __post_init__(self):
        self.peft_type = PeftType.MSLORA


class LoraLayer(BaseTunerLayer):
    def __init__(self, in_features: int, out_features: int, **kwargs):
        self.r = {}
        self.lora_alpha = {}
        self.scaling = {}
        self.lora_dropout = nn.ModuleDict({})
        self.lora_A = nn.ModuleDict({})
        self.lora_B = nn.ModuleDict({})
        # For Embedding layer
        self.lora_embedding_A = nn.ParameterDict({})
        self.lora_embedding_B = nn.ParameterDict({})
        # Mark the weight as unmerged
        self.merged = False
        self.disable_adapters = False
        self.in_features = in_features
        self.out_features = out_features
        self.kwargs = kwargs

    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights):
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        # Actual trainable parameters
        if r > 0:
            self.lora_A.update(nn.ModuleDict({adapter_name: nn.Linear(self.in_features, r, bias=False)}))
            self.lora_B.update(nn.ModuleDict({adapter_name: nn.Linear(r, self.out_features, bias=False)}))
            self.scaling[adapter_name] = lora_alpha / r
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)
        self.to(self.weight.device)

    def update_layer_conv2d(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights):
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        # Actual trainable parameters
        if r > 0:
            kernel_size = self.kwargs["kernel_size"]
            stride = self.kwargs["stride"]
            padding = self.kwargs["padding"]
            self.lora_A.update(
                nn.ModuleDict({adapter_name: nn.Conv2d(self.in_features, r, kernel_size, stride, padding, bias=False)})
            )
            self.lora_B.update(
                nn.ModuleDict({adapter_name: nn.Conv2d(r, self.out_features, (1, 1), (1, 1), bias=False)})
            )
            self.scaling[adapter_name] = lora_alpha / r
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)
        self.to(self.weight.device)

    def update_layer_embedding(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights):
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        # Actual trainable parameters
        if r > 0:
            weight_A = torch.randn((r, self.in_features), dtype=self.weight.dtype, device=self.weight.device)
            weight_B = torch.randn((self.out_features, r), dtype=self.weight.dtype, device=self.weight.device)
            self.lora_embedding_A.update(nn.ParameterDict({adapter_name: nn.Parameter(weight_A)}))
            self.lora_embedding_B.update(nn.ParameterDict({adapter_name: nn.Parameter(weight_B)}))
            self.scaling[adapter_name] = lora_alpha / r
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)
        self.to(self.weight.device)

    def reset_lora_parameters(self, adapter_name):
        if adapter_name in self.lora_A.keys():
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A[adapter_name].weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B[adapter_name].weight)
        if adapter_name in self.lora_embedding_A.keys():
            # initialize a the same way as the default for nn.linear and b to zero
            nn.init.zeros_(self.lora_embedding_A[adapter_name])
            nn.init.normal_(self.lora_embedding_B[adapter_name])

class MSLoraModel(BaseTuner):
    """
    Creates Low Rank Adapter (Lora) model from a pretrained transformers model.

    Args:
        model ([`~transformers.PreTrainedModel`]): The model to be adapted.
        config ([`LoraConfig`]): The configuration of the Lora model.
        adapter_name (`str`): The name of the adapter, defaults to `"default"`.

    Returns:
        `torch.nn.Module`: The Lora model.

    Example:

        ```py
        >>> from transformers import AutoModelForSeq2SeqLM
        >>> from peft import LoraModel, LoraConfig

        >>> config = LoraConfig(
        ...     task_type="SEQ_2_SEQ_LM",
        ...     r=8,
        ...     lora_alpha=32,
        ...     target_modules=["q", "v"],
        ...     lora_dropout=0.01,
        ... )

        >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
        >>> lora_model = LoraModel(model, config, "default")
        ```

        ```py
        >>> import transformers
        >>> from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_int8_training

        >>> target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "fc_in", "fc_out", "wte"]
        >>> config = LoraConfig(
        ...     r=4, lora_alpha=16, target_modules=target_modules, lora_dropout=0.1, bias="none", task_type="CAUSAL_LM"
        ... )

        >>> model = transformers.GPTJForCausalLM.from_pretrained(
        ...     "kakaobrain/kogpt",
        ...     revision="KoGPT6B-ryan1.5b-float16",  # or float32 version: revision=KoGPT6B-ryan1.5b
        ...     pad_token_id=tokenizer.eos_token_id,
        ...     use_cache=False,
        ...     device_map={"": rank},
        ...     torch_dtype=torch.float16,
        ...     load_in_8bit=True,
        ... )
        >>> model = prepare_model_for_int8_training(model)
        >>> lora_model = get_peft_model(model, config)
        ```

    **Attributes**:
        - **model** ([`~transformers.PreTrainedModel`]) -- The model to be adapted.
        - **peft_config** ([`LoraConfig`]): The configuration of the Lora model.
    """

    def __init__(self, model, config, adapter_name) -> None:
        super().__init__(model, config, adapter_name)

    def _check_new_adapter_config(self, config: MSLoraConfig) -> None:
        """
        A helper method to check the config when a new adapter is being added.

        Raise a ValueError if there is something wrong with the config or if it conflicts with existing adapters.

        """
        # TODO: there should be a check if any of the existing adapters actually has bias != "none", or else the check
        # does not fully correspond to the error message.
        if (len(self.peft_config) > 1) and (config.bias != "none"):
            raise ValueError(
                f"{self.__class__.__name__} supports only 1 adapter with bias. When using multiple adapters, "
                "set bias to 'none' for all adapters."
            )

    @staticmethod
    def _check_target_module_exists(lora_config, key):
        if isinstance(lora_config.target_modules, str):
            target_module_found = re.fullmatch(lora_config.target_modules, key)
        else:
            target_module_found = any(
                re.match(f".*\.{target_key}$", key) for target_key in lora_config.target_modules
            ) or any(target_key == key for target_key in lora_config.target_modules)
            is_using_layer_indexes = getattr(lora_config, "layers_to_transform", None) is not None
            layer_indexing_pattern = getattr(lora_config, "layers_pattern", None)

            if is_using_layer_indexes and target_module_found:
                layers_pattern = COMMON_LAYERS_PATTERN if layer_indexing_pattern is None else layer_indexing_pattern
                layers_pattern = [layers_pattern] if isinstance(layers_pattern, str) else layers_pattern

                for pattern in layers_pattern:
                    layer_index = re.match(f".*.{pattern}\.(\d+)\.*", key)
                    if layer_index is not None:
                        layer_index = int(layer_index.group(1))
                        if isinstance(lora_config.layers_to_transform, int):
                            target_module_found = layer_index == lora_config.layers_to_transform
                        else:
                            target_module_found = layer_index in lora_config.layers_to_transform

                        break
                    else:
                        target_module_found = False
        return target_module_found

    def _create_and_replace(
        self,
        lora_config,
        adapter_name,
        target,
        target_name,
        parent,
        **optionnal_kwargs,
    ):
        bias = hasattr(target, "bias") and target.bias is not None
        kwargs = {
            "r": lora_config.r,
            "lora_alpha": lora_config.lora_alpha,
            "lora_dropout": lora_config.lora_dropout,
            "fan_in_fan_out": lora_config.fan_in_fan_out,
            "init_lora_weights": lora_config.init_lora_weights,
        }
        kwargs["loaded_in_8bit"] = optionnal_kwargs.pop("loaded_in_8bit", False)
        kwargs["loaded_in_4bit"] = optionnal_kwargs.pop("loaded_in_4bit", False)
        kwargs["bias"] = bias

        quantization_config = get_quantization_config(self.model, method="gptq")
        if quantization_config is not None:
            kwargs["gptq_quantization_config"] = quantization_config

        # TODO: better deal with that
        if isinstance(target, LoraLayer) and isinstance(target, torch.nn.Conv2d):
            target.update_layer_conv2d(
                adapter_name,
                lora_config.r,
                lora_config.lora_alpha,
                lora_config.lora_dropout,
                lora_config.init_lora_weights,
            )
        elif isinstance(target, LoraLayer) and isinstance(target, torch.nn.Embedding):
            target.update_layer_embedding(
                adapter_name,
                lora_config.r,
                lora_config.lora_alpha,
                lora_config.lora_dropout,
                lora_config.init_lora_weights,
            )

        elif isinstance(target, LoraLayer):
            target.update_layer(
                adapter_name,
                lora_config.r,
                lora_config.lora_alpha,
                lora_config.lora_dropout,
                lora_config.init_lora_weights,
            )
        else:
            new_module = self._create_new_module(lora_config, adapter_name, target, **kwargs)
            self._replace_module(parent, target_name, new_module, target)

    @staticmethod
    def _replace_module(parent, child_name, new_module, child):
        setattr(parent, child_name, new_module)
        new_module.weight = child.weight
        if hasattr(child, "bias"):
            if child.bias is not None:
                new_module.bias = child.bias

        if getattr(child, "state", None) is not None:
            new_module.state = child.state
            new_module.to(child.weight.device)

        # dispatch to correct device
        for name, module in new_module.named_modules():
            if "lora_" in name:
                module.to(child.weight.device)
            if "ranknum" in name:
                module.to(child.weight.device)

    def _mark_only_adapters_as_trainable(self) -> None:
        active_adapter = self._get_active_adapter()
        bias = self.peft_config[active_adapter].bias

        for n, p in self.model.named_parameters():
            if "lora_" not in n:
                p.requires_grad = False
        if bias == "none":
            return
        elif bias == "all":
            for n, p in self.model.named_parameters():
                if "bias" in n:
                    p.requires_grad = True
        elif bias == "lora_only":
            for m in self.model.modules():
                if isinstance(m, LoraLayer) and hasattr(m, "bias") and m.bias is not None:
                    m.bias.requires_grad = True
        else:
            raise NotImplementedError

    @staticmethod
    def _create_new_module(lora_config, adapter_name, target, **kwargs):
        gptq_quantization_config = kwargs.get("gptq_quantization_config", None)
        AutoGPTQQuantLinear = get_auto_gptq_quant_linear(gptq_quantization_config)

        loaded_in_8bit = kwargs.pop("loaded_in_8bit", False)
        loaded_in_4bit = kwargs.pop("loaded_in_4bit", False)
        bias = kwargs.pop("bias", False)

        if loaded_in_8bit and isinstance(target, bnb.nn.Linear8bitLt):
            eightbit_kwargs = kwargs.copy()
            eightbit_kwargs.update(
                {
                    "has_fp16_weights": target.state.has_fp16_weights,
                    "memory_efficient_backward": target.state.memory_efficient_backward,
                    "threshold": target.state.threshold,
                    "index": target.index,
                }
            )
            new_module = Linear8bitLt(
                adapter_name, target.in_features, target.out_features, bias=bias, **eightbit_kwargs
            )
        elif loaded_in_4bit and is_bnb_4bit_available() and isinstance(target, bnb.nn.Linear4bit):
            fourbit_kwargs = kwargs.copy()
            fourbit_kwargs.update(
                {
                    "compute_dtype": target.compute_dtype,
                    "compress_statistics": target.weight.compress_statistics,
                    "quant_type": target.weight.quant_type,
                }
            )
            new_module = Linear4bit(adapter_name, target.in_features, target.out_features, bias=bias, **fourbit_kwargs)
        elif AutoGPTQQuantLinear is not None and isinstance(target, AutoGPTQQuantLinear):
            new_module = QuantLinear(adapter_name, target, **kwargs)
            target.weight = target.qweight
        elif isinstance(target, torch.nn.Embedding):
            embedding_kwargs = kwargs.copy()
            embedding_kwargs.pop("fan_in_fan_out", None)
            in_features, out_features = target.num_embeddings, target.embedding_dim
            new_module = Embedding(adapter_name, in_features, out_features, **embedding_kwargs)
        elif isinstance(target, torch.nn.Conv2d):
            out_channels, in_channels = target.weight.size()[:2]
            kernel_size = target.weight.size()[2:]
            stride = target.stride
            padding = target.padding
            new_module = Conv2d(adapter_name, in_channels, out_channels, kernel_size, stride, padding, **kwargs)
        else:
            if isinstance(target, torch.nn.Linear):
                in_features, out_features = target.in_features, target.out_features
                if kwargs["fan_in_fan_out"]:
                    warnings.warn(
                        "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                        "Setting fan_in_fan_out to False."
                    )
                    kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
                new_module = Linear(adapter_name, in_features, out_features, bias=bias, **kwargs)
            elif isinstance(target, Conv1D):
                in_features, out_features = (
                    target.weight.ds_shape if hasattr(target.weight, "ds_shape") else target.weight.shape
                )
                kwargs["is_target_conv_1d_layer"] = True
                if not kwargs["fan_in_fan_out"]:
                    warnings.warn(
                        "fan_in_fan_out is set to False but the target module is `Conv1D`. "
                        "Setting fan_in_fan_out to True."
                    )
                    kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = True
                new_module = Linear(adapter_name, in_features, out_features, bias=bias, **kwargs)
            else:
                raise ValueError(
                    f"Target module {target} is not supported. "
                    f"Currently, only `torch.nn.Linear` and `Conv1D` are supported."
                )
                new_module = Linear(adapter_name, in_features, out_features, bias=bias, **kwargs)

        return new_module

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

    def get_peft_config_as_dict(self, inference: bool = False):
        config_dict = {}
        for key, value in self.peft_config.items():
            config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(value).items()}
            if inference:
                config["inference_mode"] = True
        config_dict[key] = config
        return config

    def _set_adapter_layers(self, enabled=True):
        for module in self.model.modules():
            if isinstance(module, LoraLayer):
                module.disable_adapters = False if enabled else True
            elif isinstance(module, ModulesToSaveWrapper):
                module.disable_adapters = False if enabled else True

    def enable_adapter_layers(self):
        self._set_adapter_layers(enabled=True)

    def _get_active_adapter(self) -> str:
        active_adapter = None
        for module in self.model.modules():
            if isinstance(module, LoraLayer):
                active_adapter = module.active_adapter

        if active_adapter is None:
            raise ValueError(
                "Something went wrong, no active adapter could be found, please report the issue on GitHub"
            )
        return active_adapter

    def disable_adapter_layers(self):
        active_adapter = self._get_active_adapter()
        val = self.peft_config[active_adapter].bias
        if val != "none":
            msg = (
                f"Careful, disabling adapter layers with bias configured to be '{val}' does not produce the same "
                "output as the the base model would without adaption."
            )
            warnings.warn(msg)
        self._set_adapter_layers(enabled=False)

    def set_adapter(self, adapter_name):
        for module in self.model.modules():
            if isinstance(module, LoraLayer):
                if module.merged:
                    warnings.warn("Adapter cannot be set when the model is merged. Unmerging the model first.")
                    module.unmerge()
                module.active_adapter = adapter_name

    @staticmethod
    def _prepare_adapter_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_MSLORA_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = TRANSFORMERS_MODELS_TO_MSLORA_TARGET_MODULES_MAPPING[model_config["model_type"]]
        return peft_config

    def _unload_and_optionally_merge(self, merge=True, progressbar: bool = False):
        if merge:
            if getattr(self.model, "is_loaded_in_8bit", False) or getattr(self.model, "is_loaded_in_4bit", False):
                raise ValueError("Cannot merge LORA layers when the model is loaded in 8-bit mode")
            if getattr(self.model, "quantization_method", None) == "gptq":
                raise ValueError("Cannot merge LORA layers when the model is gptq quantized")

        key_list = [key for key, _ in self.model.named_modules() if "lora" not in key]
        desc = "Unloading " + ("and merging " if merge else "") + "model"
        for key in tqdm(key_list, disable=not progressbar, desc=desc):
            try:
                parent, target, target_name = _get_submodules(self.model, key)
            except AttributeError:
                continue
            if isinstance(target, LoraLayer):
                if isinstance(target, nn.Embedding):
                    new_module = torch.nn.Embedding(target.in_features, target.out_features)
                elif isinstance(target, nn.Conv2d):
                    new_module = torch.nn.Conv2d(
                        target.in_channels,
                        target.out_channels,
                        kernel_size=target.kernel_size,
                        stride=target.stride,
                        padding=target.padding,
                        dilation=target.dilation,
                    )
                else:
                    bias = target.bias is not None
                    if getattr(target, "is_target_conv_1d_layer", False):
                        new_module = Conv1D(target.out_features, target.in_features)
                    else:
                        new_module = torch.nn.Linear(target.in_features, target.out_features, bias=bias)
                if merge:
                    target.merge()
                self._replace_module(parent, target_name, new_module, target)

            # save any additional trainable modules part of `modules_to_save`
            if isinstance(target, ModulesToSaveWrapper):
                setattr(parent, target_name, target.modules_to_save[target.active_adapter])

        return self.model

    def add_weighted_adapter(
        self,
        adapters,
        weights,
        adapter_name,
        combination_type="svd",
        svd_rank=None,
        svd_clamp=None,
        svd_full_matrices=True,
        svd_driver=None,
    ):
        """
        This method adds a new adapter by merging the given adapters with the given weights.

        When using the `cat` combination_type you should be aware that rank of the resulting adapter will be equal to
        the sum of all adapters ranks. So it's possible that the mixed adapter may become too big and result in OOM
        errors.

        Args:
            adapters (`list`):
                List of adapter names to be merged.
            weights (`list`):
                List of weights for each adapter.
            adapter_name (`str`):
                Name of the new adapter.
            combination_type (`str`):
                Type of merging. Can be one of [`svd`, `linear`, `cat`]. When using the `cat` combination_type you
                should be aware that rank of the resulting adapter will be equal to the sum of all adapters ranks. So
                it's possible that the mixed adapter may become too big and result in OOM errors.
            svd_rank (`int`, *optional*):
                Rank of output adapter for svd. If None provided, will use max rank of merging adapters.
            svd_clamp (`float`, *optional*):
                A quantile threshold for clamping SVD decomposition output. If None is provided, do not perform
                clamping. Defaults to None.
            svd_full_matrices (`bool`, *optional*):
                Controls whether to compute the full or reduced SVD, and consequently, the shape of the returned
                tensors U and Vh. Defaults to True.
            svd_driver (`str`, *optional*):
                Name of the cuSOLVER method to be used. This keyword argument only works when merging on CUDA. Can be
                one of [None, `gesvd`, `gesvdj`, `gesvda`]. For more info please refer to `torch.linalg.svd`
                documentation. Defaults to None.
        """

        if adapter_name in list(self.peft_config.keys()):
            return
        for adapter in adapters:
            if adapter not in list(self.peft_config.keys()):
                raise ValueError(f"Adapter {adapter} does not exist")

        # if there is only one adapter, we can only use linear merging
        combination_type = "linear" if len(adapters) == 1 else combination_type

        adapters_ranks = [self.peft_config[adapter].r for adapter in adapters]
        if combination_type == "linear":
            # all adapters ranks should be same, new rank is just this value
            if len(set(adapters_ranks)) != 1:
                raise ValueError("All adapters must have the same r value when using `linear` combination_type")
            new_rank = adapters_ranks[0]
        elif combination_type == "cat":
            # adapters ranks may be different, new rank is sum of all ranks
            # be careful, because output adapter rank may be really big if mixing a lot of adapters
            new_rank = sum(adapters_ranks)
        elif combination_type == "svd":
            # new rank is the max of all ranks of the adapters if not provided
            new_rank = svd_rank or max(adapters_ranks)
        else:
            raise ValueError(f"Invalid combination_type: {combination_type}")

        self.peft_config[adapter_name] = replace(self.peft_config[adapters[0]], r=new_rank, lora_alpha=new_rank)
        self.inject_adapter(self.model, adapter_name)

        # Do we really need that?
        _freeze_adapter(self.model, adapter_name)

        key_list = [key for key, _ in self.model.named_modules() if "lora" not in key]
        for key in key_list:
            _, target, _ = _get_submodules(self.model, key)
            if isinstance(target, LoraLayer):
                if adapter_name in target.lora_A:
                    target_lora_A = target.lora_A[adapter_name].weight
                    target_lora_B = target.lora_B[adapter_name].weight
                elif adapter_name in target.lora_embedding_A:
                    target_lora_A = target.lora_embedding_A[adapter_name]
                    target_lora_B = target.lora_embedding_B[adapter_name]

                target_lora_A.data = target_lora_A.data * 0.0
                target_lora_B.data = target_lora_B.data * 0.0
                if combination_type == "linear":
                    for adapter, weight in zip(adapters, weights):
                        if adapter in target.lora_A:
                            current_adapter_lora_A = target.lora_A[adapter].weight
                            current_adapter_lora_B = target.lora_B[adapter].weight
                        elif adapter in target.lora_embedding_A:
                            current_adapter_lora_A = target.lora_embedding_A[adapter]
                            current_adapter_lora_B = target.lora_embedding_B[adapter]
                        target_lora_A.data += current_adapter_lora_A.data * weight * target.scaling[adapter]
                        target_lora_B.data += current_adapter_lora_B.data
                elif combination_type == "cat":
                    loras_A, loras_B = [], []
                    for adapter, weight in zip(adapters, weights):
                        if adapter in target.lora_A:
                            current_adapter_lora_A = target.lora_A[adapter].weight
                            current_adapter_lora_B = target.lora_B[adapter].weight
                        elif adapter in target.lora_embedding_A:
                            current_adapter_lora_A = target.lora_embedding_A[adapter]
                            current_adapter_lora_B = target.lora_embedding_B[adapter]
                        loras_A.append(current_adapter_lora_A.data * weight * target.scaling[adapter])
                        loras_B.append(current_adapter_lora_B.data)
                    torch.cat(loras_A, dim=0, out=target_lora_A.data)
                    torch.cat(loras_B, dim=1, out=target_lora_B.data)
                elif combination_type == "svd":
                    target_lora_A.data, target_lora_B.data = self._svd_weighted_adapter(
                        adapters,
                        weights,
                        new_rank,
                        target,
                        target_lora_A,
                        target_lora_B,
                        svd_clamp,
                        full_matrices=svd_full_matrices,
                        driver=svd_driver,
                    )

    def _svd_weighted_adapter(
        self,
        adapters,
        weights,
        new_rank,
        target,
        target_lora_A,
        target_lora_B,
        clamp=None,
        full_matrices=True,
        driver=None,
    ):
        delta_weight = weights[0] * target.get_delta_weight(adapters[0])
        for adapter, weight in zip(adapters[1:], weights[1:]):
            delta_weight += weight * target.get_delta_weight(adapter)
        conv2d = isinstance(target, Conv2d)
        if conv2d:
            conv2d_1x1 = target.weight.size()[2:4] == (1, 1)
            if not conv2d_1x1:
                delta_weight = delta_weight.flatten(start_dim=1)
            else:
                delta_weight = delta_weight.squeeze()
        if hasattr(target, "fan_in_fan_out") and target.fan_in_fan_out:
            delta_weight = delta_weight.T

        # based on https://github.com/kohya-ss/sd-scripts/blob/main/networks/svd_merge_lora.py#L114-L131
        U, S, Vh = torch.linalg.svd(delta_weight, full_matrices=full_matrices, driver=driver)
        U = U[:, :new_rank]
        S = S[:new_rank]
        U = U @ torch.diag(S)
        Vh = Vh[:new_rank, :]
        if clamp is not None:
            dist = torch.cat([U.flatten(), Vh.flatten()])
            hi_val = torch.quantile(dist, clamp)
            low_val = -hi_val
            U = U.clamp(low_val, hi_val)
            Vh = Vh.clamp(low_val, hi_val)
        if conv2d:
            U = U.reshape(target_lora_B.data.shape)
            Vh = Vh.reshape(target_lora_A.data.shape)
        return Vh, U

    def delete_adapter(self, adapter_name):
        """
        Deletes an existing adapter.

        Args:
            adapter_name (str): Name of the adapter to be deleted.
        """
        if adapter_name not in list(self.peft_config.keys()):
            raise ValueError(f"Adapter {adapter_name} does not exist")
        del self.peft_config[adapter_name]
        key_list = [key for key, _ in self.model.named_modules() if "lora" not in key]
        for key in key_list:
            _, target, _ = _get_submodules(self.model, key)
            if isinstance(target, LoraLayer):
                for attr in [
                    "r",
                    "lora_alpha",
                    "scaling",
                    "lora_A",
                    "lora_B",
                    "lora_embedding_A",
                    "lora_embedding_B",
                    "lora_dropout",
                ]:
                    if adapter_name in getattr(target, attr):
                        getattr(target, attr).pop(adapter_name)
                if target.active_adapter == adapter_name:
                    resetting_active_adapter = list(self.peft_config.keys())[0]
                    warnings.warn(
                        f"Adapter {adapter_name} was active which is now deleted. Setting active adapter to {resetting_active_adapter}. "
                    )
                    target.active_adapter = resetting_active_adapter

    def merge_and_unload(self, progressbar: bool = False):
        r"""
        This method merges the LoRa layers into the base model. This is needed if someone wants to use the base model
        as a standalone model.

        Args:
            progressbar (bool): whether to show a progressbar indicating the unload and merge process

        Example:

        ```py
        >>> from transformers import AutoModelForCausalLM
        >>> from peft import PeftModel

        >>> base_model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-40b")
        >>> peft_model_id = "smangrul/falcon-40B-int4-peft-lora-sfttrainer-sample"
        >>> model = PeftModel.from_pretrained(base_model, peft_model_id)
        >>> merged_model = model.merge_and_unload()
        ```
        """
        return self._unload_and_optionally_merge(progressbar=progressbar)

    def unload(self):
        """
        Gets back the base model by removing all the lora modules without merging. This gives back the original base
        model.
        """
        return self._unload_and_optionally_merge(merge=False)


# Below code is based on https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# and modified to work with PyTorch FSDP


#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------


class Linear(nn.Linear, LoraLayer):
    # Lora implemented in a dense layer
    def __init__(
        self,
        adapter_name: str,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        **kwargs,
    ):
        init_lora_weights = kwargs.pop("init_lora_weights", True)

        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoraLayer.__init__(self, in_features=in_features, out_features=out_features)
        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False

        self.fan_in_fan_out = fan_in_fan_out
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

        nn.Linear.reset_parameters(self)
        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
        self.active_adapter = adapter_name
        self.is_target_conv_1d_layer = is_target_conv_1d_layer

    def merge(self):
        if self.active_adapter not in self.lora_A.keys():
            return
        if self.merged:
            warnings.warn("Already merged. Nothing to do.")
            return
        if self.r[self.active_adapter] > 0:
            self.weight.data += self.get_delta_weight(self.active_adapter)
            self.merged = True

    def unmerge(self):
        if self.active_adapter not in self.lora_A.keys():
            return
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        if self.r[self.active_adapter] > 0:
            self.weight.data -= self.get_delta_weight(self.active_adapter)
            self.merged = False

    def get_delta_weight(self, adapter):
        return (
            transpose(
                self.lora_B[adapter].weight @ self.lora_A[adapter].weight,
                self.fan_in_fan_out,
            )
            * self.scaling[adapter]
        )

    def forward(self, x: torch.Tensor):
        previous_dtype = x.dtype
        # in_time = time.time()
        if self.active_adapter not in self.lora_A.keys():
            return F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        if self.disable_adapters:
            if self.r[self.active_adapter] > 0 and self.merged:
                self.unmerge()
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        elif self.r[self.active_adapter] > 0 and not self.merged:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

            x = x.to(self.lora_A[self.active_adapter].weight.dtype)

            result += (
                self.lora_B[self.active_adapter](
                    self.lora_A[self.active_adapter](self.lora_dropout[self.active_adapter](x))
                )
                * self.scaling[self.active_adapter]
            )
            # out_time = time.time()
            # print("Time for lora: ", out_time - in_time)
        else:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        result = result.to(previous_dtype)

        return result


class Embedding(nn.Embedding, LoraLayer):
    # LoRA implemented in a Embedding layer
    def __init__(
        self,
        adapter_name: str,
        num_embeddings: int,
        embedding_dim: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        **kwargs,
    ):
        init_lora_weights = kwargs.pop("init_lora_weights", True)

        nn.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)
        LoraLayer.__init__(self, in_features=num_embeddings, out_features=embedding_dim)

        self.weight.requires_grad = False

        nn.Embedding.reset_parameters(self)
        self.update_layer_embedding(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
        self.active_adapter = adapter_name

    def unmerge(self):
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        if self.r[self.active_adapter] > 0:
            self.weight.data -= self.get_delta_weight(self.active_adapter)
            self.merged = False

    def merge(self):
        if self.merged:
            warnings.warn("Already merged. Nothing to do.")
            return
        if self.r[self.active_adapter] > 0:
            self.weight.data += self.get_delta_weight(self.active_adapter)
            self.merged = True

    def get_delta_weight(self, adapter):
        return transpose(self.lora_embedding_B[adapter] @ self.lora_embedding_A[adapter], True) * self.scaling[adapter]

    def forward(self, x: torch.Tensor):
        if self.disable_adapters:
            if self.r[self.active_adapter] > 0 and self.merged:
                self.unmerge()
            return nn.Embedding.forward(self, x)

        elif self.r[self.active_adapter] > 0 and not self.merged:
            result = nn.Embedding.forward(self, x)
            if self.r[self.active_adapter] > 0:
                after_A = F.embedding(
                    x,
                    self.lora_embedding_A[self.active_adapter].T,
                    self.padding_idx,
                    self.max_norm,
                    self.norm_type,
                    self.scale_grad_by_freq,
                    self.sparse,
                )
                result += (after_A @ self.lora_embedding_B[self.active_adapter].T) * self.scaling[self.active_adapter]
            return result
        else:
            return nn.Embedding.forward(self, x)


class Conv2d(nn.Conv2d, LoraLayer):
    # Lora implemented in a conv2d layer
    def __init__(
        self,
        adapter_name: str,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int]],
        stride: Union[int, Tuple[int]] = 1,
        padding: Union[int, Tuple[int]] = 0,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        **kwargs,
    ):
        init_lora_weights = kwargs.pop("init_lora_weights", True)

        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, stride, padding)
        LoraLayer.__init__(
            self,
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False

        nn.Conv2d.reset_parameters(self)
        self.update_layer_conv2d(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
        self.active_adapter = adapter_name

    def merge(self):
        if self.active_adapter not in self.lora_A.keys():
            return
        if self.merged:
            warnings.warn("Already merged. Nothing to do.")
            return
        if self.r[self.active_adapter] > 0:
            self.weight.data += self.get_delta_weight(self.active_adapter)
            self.merged = True

    def unmerge(self):
        if self.active_adapter not in self.lora_A.keys():
            return
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        if self.r[self.active_adapter] > 0:
            self.weight.data -= self.get_delta_weight(self.active_adapter)
            self.merged = False

    def get_delta_weight(self, adapter):
        # https://github.com/bmaltais/kohya_ss/blob/feb6728762a8f463d15ba936d189d4c3abfaa1ab/networks/lora.py#L117
        if self.weight.size()[2:4] == (1, 1):
            # conv2d 1x1
            return (
                self.lora_B[adapter].weight.squeeze(3).squeeze(2) @ self.lora_A[adapter].weight.squeeze(3).squeeze(2)
            ).unsqueeze(2).unsqueeze(3) * self.scaling[adapter]
        else:
            # conv2d 3x3
            return (
                F.conv2d(
                    self.lora_A[adapter].weight.permute(1, 0, 2, 3),
                    self.lora_B[adapter].weight,
                ).permute(1, 0, 2, 3)
                * self.scaling[adapter]
            )

    def forward(self, x: torch.Tensor):
        previous_dtype = x.dtype

        if self.active_adapter not in self.lora_A.keys():
            return F.conv2d(
                x,
                self.weight,
                bias=self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )
        if self.disable_adapters:
            if self.r[self.active_adapter] > 0 and self.merged:
                self.unmerge()
            result = F.conv2d(
                x,
                self.weight,
                bias=self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )
        elif self.r[self.active_adapter] > 0 and not self.merged:
            result = F.conv2d(
                x,
                self.weight,
                bias=self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )

            x = x.to(self.lora_A[self.active_adapter].weight.dtype)

            result += (
                self.lora_B[self.active_adapter](
                    self.lora_A[self.active_adapter](self.lora_dropout[self.active_adapter](x))
                )
                * self.scaling[self.active_adapter]
            )
        else:
            result = F.conv2d(
                x,
                self.weight,
                bias=self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )

        result = result.to(previous_dtype)

        return result


class LlamaLoRALayer(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        """
        LoRA module, consisting of multiple adapters, each adapter contains two low-rank linear layers.

        Args:
            in_features (int): Input feature dimension.
            out_features (int): Output feature dimension.
        """
        super(LlamaLoRALayer, self).__init__()
        self.r = {}
        self.lora_alpha = {}
        self.scaling = {}
        self.lora_dropout = nn.ModuleDict({})
        self.lora_A = nn.ModuleDict({})
        self.lora_B = nn.ModuleDict({})
        self.merged = False
        self.disable_adapters = False
        self.in_features = in_features
        self.out_features = out_features
        self.input_cache = {}

    def update_layer(self, adapter_name: str, r: int, lora_alpha: float, lora_dropout: float, init_lora_weights: bool = True):
        """
        Add a new LoRA adapter.

        Args:
            adapter_name (str): Adapter name for unique identification.
            r (int): Low-rank dimension.
            lora_alpha (float): Scaling factor.
            lora_dropout (float): Dropout probability.
            init_lora_weights (bool): Whether to initialize the weights.
        """
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update({adapter_name: lora_dropout_layer})
        # Add low-rank linear layers
        if r > 0:
            self.lora_A.update({adapter_name: nn.Linear(self.in_features, r, bias=False)})
            self.lora_B.update({adapter_name: nn.Linear(r, self.out_features, bias=False)})
            self.scaling[adapter_name] = lora_alpha / r
        if init_lora_weights and r > 0:
            self.reset_lora_parameters(adapter_name)
        # Move the LoRA layers to the same device as the main module
        if len(self.lora_A) > 0:
            device = next(self.lora_A[adapter_name].parameters()).device
            self.lora_A.to(device)
            self.lora_B.to(device)
            self.lora_dropout.to(device)

    def reset_lora_parameters(self, adapter_name: str):
        """
        Initialize the parameters of the LoRA layer.

        Args:
            adapter_name (str): Adapter name.
        """
        if adapter_name in self.lora_A:
            nn.init.kaiming_uniform_(self.lora_A[adapter_name].weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B[adapter_name].weight)

class LoRAParallelEncoder(LoraLayer, nn.Module):
    def __init__(
        self,
        base_encoder,
        lora_parallel_schedule,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        init_lora_weights=True,
        activation_fn=nn.ReLU(),  # Default activation function
        init_activate_functions=False,
        init_noise_weights=False,  # Whether to initialize the random noise matrix W
        fan_in_fan_out=False,
        **kwargs
    ):
        nn.Module.__init__(self)
        self.weight = nn.Parameter(torch.zeros(1, 1), requires_grad=False)
        self.base_encoder = base_encoder
        self.layers = base_encoder.layer
        self.in_features = self.layers[0].attention.self.query.in_features
        self.out_features = self.layers[0].attention.self.query.out_features

        LoraLayer.__init__(self, in_features=self.in_features, out_features=self.out_features, **kwargs)

        # Freeze the parameters of the original layers
        for layer in self.layers:
            for param in layer.parameters():
                param.requires_grad = False

        self.adapter_weights = nn.ParameterDict()
        self.lora_schedules = {}
        # Add an activation function dictionary
        self.lora_activations = nn.ModuleDict() if init_activate_functions else None
        self.adapter_noises = nn.ParameterDict() if init_noise_weights else None
        for start_idx, end_idx, adapter_name in lora_parallel_schedule:
            self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
            self.adapter_weights.update({
                adapter_name: nn.Parameter(torch.zeros(self.out_features, self.in_features), requires_grad=False)
            })
            # Initialize the activation function. You can choose other activation functions as needed.
            if init_activate_functions:
                self.lora_activations.update({
                    adapter_name: activation_fn
                })
            if self.adapter_noises is not None:
                # noise_w = nn.Parameter(torch.randn(self.out_features, self.in_features), requires_grad=False)
                noise_w = nn.Parameter(torch.zeros(self.out_features, self.in_features), requires_grad=False)
                self.adapter_noises.update({adapter_name: noise_w})
            self.lora_schedules[adapter_name] = {
                "start_idx": start_idx,
                "end_idx": end_idx,
                "merged": False
            }

        self.disable_adapters = False
        self.input_cache = {}
        self.init_noise_weights = init_noise_weights
        self.init_activate_functions = init_activate_functions

    def merge(self, adapter_name):
        if adapter_name not in self.lora_A.keys():
            return
        if self.lora_schedules[adapter_name]["merged"]:
            warnings.warn("Already merged. Nothing to do.")
            return
        if self.r[adapter_name] > 0:
            delta_weight = (self.lora_B[adapter_name].weight @ self.lora_A[adapter_name].weight) * self.scaling[adapter_name]
            self.adapter_weights[adapter_name].data += delta_weight
            self.lora_schedules[adapter_name]["merged"] = True

    def unmerge(self, adapter_name):
        if adapter_name not in self.lora_A.keys():
            return
        if not self.lora_schedules[adapter_name]["merged"]:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        if self.r[adapter_name] > 0:
            delta_weight = (self.lora_B[adapter_name].weight @ self.lora_A[adapter_name].weight) * self.scaling[adapter_name]
            self.adapter_weights[adapter_name].data -= delta_weight
            self.lora_schedules[adapter_name]["merged"] = False

    def merge_two_floors(self, adapter_name1, adapter_name2, new_adapter_name, init_lora_weights=False):
        sched1 = self.lora_schedules[adapter_name1]
        sched2 = self.lora_schedules[adapter_name2]

        new_start_idx = min(sched1["start_idx"], sched2["start_idx"])
        new_end_idx = max(sched1["end_idx"], sched2["end_idx"])

        # Assume that parameters such as r and lora_alpha are the same as those of adapter_name1.
        r_val = self.r[adapter_name1]
        lora_alpha_val = self.lora_alpha[adapter_name1]
        # lora_dropout_val is an nn.Dropout or Identity, obtained through self.lora_dropout[adapter_name1].
        # If you need to get the probability p, you can use (self.lora_dropout[adapter_name1].p if isinstance(...) else 0.0).
        if isinstance(self.lora_dropout[adapter_name1], nn.Dropout):
            lora_dropout_p = self.lora_dropout[adapter_name1].p
        else:
            lora_dropout_p = 0.0

        # Create a new adapter.
        self.update_layer(new_adapter_name, r_val, lora_alpha_val, lora_dropout_p, init_lora_weights)

        # Copy the weights of adapter_name1 to the new adapter.
        self.lora_A[new_adapter_name].weight.data.copy_(self.lora_A[adapter_name1].weight.data)
        self.lora_B[new_adapter_name].weight.data.copy_(self.lora_B[adapter_name1].weight.data)
        self.scaling[new_adapter_name] = self.scaling[adapter_name1]

        self.adapter_weights.update({
            new_adapter_name: nn.Parameter(torch.zeros(self.out_features, self.in_features), requires_grad=False)
        })
        self.adapter_weights[new_adapter_name].data.copy_(self.adapter_weights[adapter_name1].data)

        if self.init_activate_functions:
            # Initialize the activation function. Choose the same activation function or change it as needed.
            self.lora_activations.update({
                new_adapter_name: self.lora_activations[adapter_name1]
            })
        if self.init_noise_weights and self.adapter_noises is not None:
            # noise_w = nn.Parameter(torch.randn(self.out_features, self.in_features), requires_grad=False)
            noise_w = nn.Parameter(torch.zeros(self.out_features, self.in_features), requires_grad=False)
            self.adapter_noises.update({new_adapter_name: noise_w})

        # Delete the old adapters.
        for old_adapter in [adapter_name1, adapter_name2]:
            del self.lora_A[old_adapter], self.lora_B[old_adapter], self.r[old_adapter], self.lora_alpha[old_adapter], self.scaling[old_adapter], self.lora_dropout[old_adapter], self.adapter_weights[old_adapter]
            del self.lora_schedules[old_adapter]
            if old_adapter in self.input_cache:
                del self.input_cache[old_adapter]
            # Also delete the activation function.
            if self.init_activate_functions and old_adapter in self.lora_activations:
                del self.lora_activations[old_adapter]
            if self.init_noise_weights and self.adapter_noises is not None and old_adapter in self.adapter_noises:
                del self.adapter_noises[old_adapter]

        self.lora_schedules[new_adapter_name] = {
            "start_idx": new_start_idx,
            "end_idx": new_end_idx,
            "merged": False
        }

        # Get the device of the current model.
        device = next(self.parameters()).device
        # Move the parameters of the newly created adapter to the current device.
        self.lora_A[new_adapter_name].to(device)
        self.lora_B[new_adapter_name].to(device)
        self.adapter_weights[new_adapter_name].data = self.adapter_weights[new_adapter_name].data.to(device)
        if self.init_noise_weights and self.adapter_noises is not None:
            self.adapter_noises[new_adapter_name].data = self.adapter_noises[new_adapter_name].data.to(device)

        # If lora_dropout is nn.Dropout, it needs to be ensured to be executed on the GPU. (However, nn.Dropout is generally stateless and does not require to().)
        # Just ensure that the parameters are on the correct device in the forward pass. Generally, there is no need to execute.to() on Dropout.

        # Ensure that after deleting the old adapter, the parameters of the newly generated adapter are all on the correct device.
        # There will be no CPU/GPU mismatch problem in the next forward pass.

    def delete_adapter(self, adapter_name):
        # Delete the specified adapter.
        if adapter_name in self.lora_A:
            del self.lora_A[adapter_name], self.lora_B[adapter_name], self.r[adapter_name], self.lora_alpha[adapter_name], self.scaling[adapter_name], self.lora_dropout[adapter_name], self.adapter_weights[adapter_name]
        if adapter_name in self.lora_schedules:
            del self.lora_schedules[adapter_name]
        if adapter_name in self.input_cache:
            del self.input_cache[adapter_name]
        if self.init_activate_functions and adapter_name in self.lora_activations:
            del self.lora_activations[adapter_name]
        if self.init_noise_weights and self.adapter_noises is not None and adapter_name in self.adapter_noises:
            del self.adapter_noises[adapter_name]

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        # Clear the input_cache at the beginning of each forward pass to ensure that the lora_input used in this batch matches this batch.
        self.input_cache.clear()

        current_device = hidden_states.device

        # Move the parameters of lora_A, lora_B, and adapter_weights to the current device.
        for adapter_name in self.lora_A.keys():
            if self.lora_A[adapter_name].weight.device != current_device:
                self.lora_A[adapter_name].to(current_device)
                self.lora_B[adapter_name].to(current_device)
            if adapter_name in self.adapter_weights and self.adapter_weights[adapter_name].device != current_device:
                self.adapter_weights[adapter_name].data = self.adapter_weights[adapter_name].data.to(current_device)
            if self.init_noise_weights and self.adapter_noises is not None and adapter_name in self.adapter_noises and self.adapter_noises[adapter_name].device != current_device:
                self.adapter_noises[adapter_name].data = self.adapter_noises[adapter_name].data.to(current_device)

            if self.lora_A[adapter_name].weight.requires_grad:
                self.lora_A[adapter_name].weight.retain_grad()
                # print(f"{adapter_name}----lora_A retain grad success!!!")
            if self.lora_B[adapter_name].weight.requires_grad:
                self.lora_B[adapter_name].weight.retain_grad()
                # print(f"{adapter_name}----lora_B retain grad success!!!")

        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        if self.disable_adapters:
            for adapter_name in self.lora_schedules:
                if self.lora_schedules[adapter_name]["merged"] and self.r[adapter_name] > 0:
                    self.unmerge(adapter_name)

        if past_key_values is None:
            past_key_values = [None] * len(self.layers)

        for i, layer_module in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # Before calculating this layer, if this layer is the start_idx of an adapter, cache the lora_input corresponding to this batch.
            if not self.disable_adapters:
                for adapter_name, sched in self.lora_schedules.items():
                    if i == sched["start_idx"]:
                        self.input_cache[adapter_name] = hidden_states

            layer_outputs = layer_module(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask[i] if head_mask is not None else None,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                past_key_value=past_key_values[i],
                output_attentions=output_attentions,
            )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[1 if output_attentions else 1],)

            if output_attentions:
                attn_idx = 1 if not use_cache else 2
                all_self_attentions += (layer_outputs[attn_idx],)

            if not self.disable_adapters:
                for adapter_name, sched in self.lora_schedules.items():
                    if i == sched["end_idx"]:
                        if adapter_name in self.lora_A.keys() and self.r[adapter_name] > 0:
                            if sched["merged"]:
                                delta = hidden_states @ self.adapter_weights[adapter_name].transpose(0, 1)
                                print(f"---------- {adapter_name} is merged ----------")
                                if self.init_noise_weights and self.adapter_noises is not None and adapter_name in self.adapter_noises:
                                    delta_noise = hidden_states @ self.adapter_noises[adapter_name].transpose(0, 1)
                                    delta = delta + delta_noise
                            else:
                                # Use the lora_input cached in the input_cache at the end_idx (corresponding to this batch).
                                lora_input = self.input_cache[adapter_name].to(current_device)
                                lora_input = self.lora_dropout[adapter_name](lora_input)
                                lora_input = lora_input.to(self.lora_A[adapter_name].weight.dtype)
                                delta = self.lora_B[adapter_name](self.lora_A[adapter_name](lora_input)) * self.scaling[adapter_name]
                                if self.init_noise_weights and self.adapter_noises is not None and adapter_name in self.adapter_noises:
                                    delta_noise = hidden_states @ self.adapter_noises[adapter_name].transpose(0, 1)
                                    delta = delta + delta_noise

                            if self.init_activate_functions and adapter_name in self.lora_activations:
                                delta = self.lora_activations[adapter_name](delta)

                            if hidden_states.shape != delta.shape:
                                print("Shape mismatch detected at layer:", i, "adapter:", adapter_name)
                                print("hidden_states:", hidden_states.shape)
                                print("delta:", delta.shape)
                                # Try to align the batch dimension by truncation.
                                min_bsz = min(hidden_states.size(0), delta.size(0))
                                # If hidden_states is longer than delta or delta is longer than hidden_states, truncate the larger one.
                                if hidden_states.size(0) > min_bsz:
                                    hidden_states = hidden_states[:min_bsz, ...]
                                if delta.size(0) > min_bsz:
                                    delta = delta[:min_bsz, ...]
                                print("After adjustment:")
                                print("hidden_states:", hidden_states.shape)
                                print("delta:", delta.shape)

                            hidden_states = hidden_states + delta

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, next_decoder_cache, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

class ShareLoRAParallelEncoder(LoraLayer, nn.Module):
    def __init__(
        self,
        base_encoder,
        lora_parallel_schedule,
        shared_lora_A_map=None,  # New parameter: Mapping for shared lora_A
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        init_lora_weights=True,
        activation_fn=nn.ReLU(),  # Default activation function
        init_activate_functions=False,
        init_noise_weights=False,  # Whether to initialize the random noise matrix W
        fan_in_fan_out=False,
        **kwargs
    ):
        nn.Module.__init__(self)
        self.weight = nn.Parameter(torch.zeros(1, 1), requires_grad=False)
        self.base_encoder = base_encoder
        self.layers = base_encoder.layer
        self.in_features = self.layers[0].attention.self.query.in_features
        self.out_features = self.layers[0].attention.self.query.out_features

        # Initialize LoraLayer
        LoraLayer.__init__(self, in_features=self.in_features, out_features=self.out_features, **kwargs)

        # Freeze the parameters of the original layers
        for layer in self.layers:
            for param in layer.parameters():
                param.requires_grad = False

        # Initialize the parameter dictionary
        self.adapter_weights = nn.ParameterDict()
        self.lora_schedules = {}
        self.lora_activations = nn.ModuleDict() if init_activate_functions else None  # Activation function dictionary
        self.adapter_noises = nn.ParameterDict() if init_noise_weights else None

        # Initialize the shared lora_A module
        self.shared_lora_A_map = shared_lora_A_map or {}
        self.shared_lora_A = {}
        if self.shared_lora_A_map:
            for group_name, adapter_names in self.shared_lora_A_map.items():
                # Assume all adapters in the shared group have the same r and lora_alpha
                r_val = r
                # Initialize the shared lora_A
                shared_lora_A = nn.Linear(self.in_features, r_val, bias=False)
                if init_lora_weights:
                    nn.init.kaiming_uniform_(shared_lora_A.weight, a=math.sqrt(5))
                self.shared_lora_A[group_name] = shared_lora_A

        # Initialize all adapters
        for start_idx, end_idx, adapter_name in lora_parallel_schedule:
            # Call the original update_layer method
            self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)

            # If the adapter belongs to a shared group, point to the shared lora_A
            if self.shared_lora_A_map:
                for group_name, adapter_names in self.shared_lora_A_map.items():
                    if adapter_name in adapter_names:
                        self.lora_A[adapter_name] = self.shared_lora_A[group_name]
                        break

            # Initialize adapter_weights
            self.adapter_weights.update({
                adapter_name: nn.Parameter(torch.zeros(self.out_features, self.in_features), requires_grad=False)
            })

            # Initialize the activation function
            if init_activate_functions:
                self.lora_activations.update({
                    adapter_name: activation_fn
                })

            # Initialize the noise matrix
            if self.adapter_noises is not None:
                noise_w = nn.Parameter(torch.zeros(self.out_features, self.in_features), requires_grad=False)
                self.adapter_noises.update({adapter_name: noise_w})

            # Define the schedule
            self.lora_schedules[adapter_name] = {
                "start_idx": start_idx,
                "end_idx": end_idx,
                "merged": False
            }

        self.disable_adapters = False
        self.input_cache = {}
        self.init_noise_weights = init_noise_weights
        self.init_activate_functions = init_activate_functions

        # Print the initialized adapters to ensure all adapters are correctly initialized
        print("Initialized adapters:", list(self.adapter_weights.keys()))

    def merge(self, adapter_name):
        if adapter_name not in self.lora_A.keys():
            return
        if self.lora_schedules[adapter_name]["merged"]:
            warnings.warn("Already merged. Nothing to do.")
            return
        if self.r[adapter_name] > 0:
            delta_weight = (self.lora_B[adapter_name].weight @ self.lora_A[adapter_name].weight) * self.scaling[adapter_name]
            self.adapter_weights[adapter_name].data += delta_weight
            self.lora_schedules[adapter_name]["merged"] = True

    def unmerge(self, adapter_name):
        if adapter_name not in self.lora_A.keys():
            return
        if not self.lora_schedules[adapter_name]["merged"]:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        if self.r[adapter_name] > 0:
            delta_weight = (self.lora_B[adapter_name].weight @ self.lora_A[adapter_name].weight) * self.scaling[adapter_name]
            self.adapter_weights[adapter_name].data -= delta_weight
            self.lora_schedules[adapter_name]["merged"] = False

    def merge_two_floors(self, adapter_name1, adapter_name2, new_adapter_name, init_lora_weights=False):
        # Get the schedule information of the two adapters
        sched1 = self.lora_schedules[adapter_name1]
        sched2 = self.lora_schedules[adapter_name2]

        # Calculate the start_idx and end_idx of the new adapter
        new_start_idx = min(sched1["start_idx"], sched2["start_idx"])
        new_end_idx = max(sched1["end_idx"], sched2["end_idx"])

        # Assume the new adapter has the same r and lora_alpha as adapter_name1
        r_val = self.r[adapter_name1]
        lora_alpha_val = self.lora_alpha[adapter_name1]
        lora_dropout_p = self.lora_dropout[adapter_name1].p if isinstance(self.lora_dropout[adapter_name1], nn.Dropout) else 0.0

        # Create a new adapter
        self.update_layer(new_adapter_name, r_val, lora_alpha_val, lora_dropout_p, init_lora_weights)

        # Check if the new adapter should share lora_A
        shared_group_name = None
        for group_name, adapter_names in self.shared_lora_A_map.items():
            if adapter_name1 in adapter_names and adapter_name2 in adapter_names:
                shared_group_name = group_name
                break

        if shared_group_name:
            # Point the lora_A of the new adapter to the shared lora_A
            self.lora_A[new_adapter_name] = self.shared_lora_A[shared_group_name]

        # Copy the weights of lora_A and lora_B
        if shared_group_name:
            # lora_A is shared, lora_B is independent
            self.lora_B[new_adapter_name].weight.data.copy_(self.lora_B[adapter_name1].weight.data)
            self.scaling[new_adapter_name] = self.scaling[adapter_name1]
        else:
            # If not shared, copy the lora_A and lora_B of adapter_name1
            self.lora_A[new_adapter_name].weight.data.copy_(self.lora_A[adapter_name1].weight.data)
            self.lora_B[new_adapter_name].weight.data.copy_(self.lora_B[adapter_name1].weight.data)
            self.scaling[new_adapter_name] = self.scaling[adapter_name1]

        # Initialize adapter_weights
        self.adapter_weights.update({
            new_adapter_name: nn.Parameter(torch.zeros(self.out_features, self.in_features), requires_grad=False)
        })
        self.adapter_weights[new_adapter_name].data.copy_(self.adapter_weights[adapter_name1].data)

        # Initialize the activation function
        if self.init_activate_functions:
            self.lora_activations.update({
                new_adapter_name: self.lora_activations[adapter_name1]
            })

        # Initialize the noise matrix
        if self.init_noise_weights and self.adapter_noises is not None:
            noise_w = nn.Parameter(torch.zeros(self.out_features, self.in_features), requires_grad=False)
            self.adapter_noises.update({new_adapter_name: noise_w})

        # Delete the old adapters
        for old_adapter in [adapter_name1, adapter_name2]:
            del self.lora_A[old_adapter], self.lora_B[old_adapter], self.r[old_adapter], self.lora_alpha[old_adapter], self.scaling[old_adapter], self.lora_dropout[old_adapter], self.adapter_weights[old_adapter]
            if old_adapter in self.lora_schedules:
                del self.lora_schedules[old_adapter]
            if old_adapter in self.input_cache:
                del self.input_cache[old_adapter]
            if self.init_activate_functions and old_adapter in self.lora_activations:
                del self.lora_activations[old_adapter]
            if self.init_noise_weights and self.adapter_noises is not None and old_adapter in self.adapter_noises:
                del self.adapter_noises[old_adapter]

        # Define the schedule for the new adapter
        self.lora_schedules[new_adapter_name] = {
            "start_idx": new_start_idx,
            "end_idx": new_end_idx,
            "merged": False
        }

        # Move the parameters of the new adapter to the current device
        device = next(self.parameters()).device
        self.lora_A[new_adapter_name].to(device)
        self.lora_B[new_adapter_name].to(device)
        self.adapter_weights[new_adapter_name].data = self.adapter_weights[new_adapter_name].data.to(device)
        if self.init_noise_weights and self.adapter_noises is not None:
            self.adapter_noises[new_adapter_name].data = self.adapter_noises[new_adapter_name].data.to(device)

    def delete_adapter(self, adapter_name):
        # Delete the specified adapter
        if adapter_name in self.lora_A:
            del self.lora_A[adapter_name], self.lora_B[adapter_name], self.r[adapter_name], self.lora_alpha[adapter_name], self.scaling[adapter_name], self.lora_dropout[adapter_name], self.adapter_weights[adapter_name]
        if adapter_name in self.lora_schedules:
            del self.lora_schedules[adapter_name]
        if adapter_name in self.input_cache:
            del self.input_cache[adapter_name]
        if self.init_activate_functions and adapter_name in self.lora_activations:
            del self.lora_activations[adapter_name]
        if self.init_noise_weights and self.adapter_noises is not None and adapter_name in self.adapter_noises:
            del self.adapter_noises[adapter_name]

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        # Clear the input_cache at the beginning of each forward pass to ensure that the lora_input used in this batch matches this batch
        self.input_cache.clear()

        current_device = hidden_states.device

        # Move the parameters of lora_A, lora_B, and adapter_weights to the current device
        for adapter_name in self.lora_A.keys():
            if self.lora_A[adapter_name].weight.device != current_device:
                self.lora_A[adapter_name].to(current_device)
                self.lora_B[adapter_name].to(current_device)
            if adapter_name in self.adapter_weights and self.adapter_weights[adapter_name].device != current_device:
                self.adapter_weights[adapter_name].data = self.adapter_weights[adapter_name].data.to(current_device)
            if self.init_noise_weights and self.adapter_noises is not None and adapter_name in self.adapter_noises and self.adapter_noises[adapter_name].device != current_device:
                self.adapter_noises[adapter_name].data = self.adapter_noises[adapter_name].data.to(current_device)

            if self.lora_A[adapter_name].weight.requires_grad:
                self.lora_A[adapter_name].weight.retain_grad()
            if self.lora_B[adapter_name].weight.requires_grad:
                self.lora_B[adapter_name].weight.retain_grad()

        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        if self.disable_adapters:
            for adapter_name in self.lora_schedules:
                if self.lora_schedules[adapter_name]["merged"] and self.r[adapter_name] > 0:
                    self.unmerge(adapter_name)

        if past_key_values is None:
            past_key_values = [None] * len(self.layers)

        for i, layer_module in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # Before calculating this layer, if this layer is the start_idx of an adapter, cache the lora_input corresponding to this batch
            if not self.disable_adapters:
                for adapter_name, sched in self.lora_schedules.items():
                    if i == sched["start_idx"]:
                        self.input_cache[adapter_name] = hidden_states.clone()

            # Pass through the Transformer layer
            layer_outputs = layer_module(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask[i] if head_mask is not None else None,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                past_key_value=past_key_values[i],
                output_attentions=output_attentions,
            )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[1],)

            if output_attentions:
                attn_idx = 1 if not use_cache else 2
                all_self_attentions += (layer_outputs[attn_idx],)

            # Apply the adapter
            if not self.disable_adapters:
                for adapter_name, sched in self.lora_schedules.items():
                    if i == sched["end_idx"]:
                        if adapter_name in self.lora_A.keys() and self.r[adapter_name] > 0:
                            if sched["merged"]:
                                # Merged, directly use adapter_weights
                                delta = hidden_states @ self.adapter_weights[adapter_name].transpose(0, 1)
                                print(f"---------- {adapter_name} is merged ----------")
                                if self.init_noise_weights and self.adapter_noises is not None and adapter_name in self.adapter_noises:
                                    delta_noise = hidden_states @ self.adapter_noises[adapter_name].transpose(0,1)
                                    delta = delta + delta_noise
                            else:
                                # Not merged, use the cached lora_input for calculation
                                lora_input = self.input_cache[adapter_name].to(current_device)
                                lora_input = self.lora_dropout[adapter_name](lora_input)
                                lora_input = lora_input.to(self.lora_A[adapter_name].weight.dtype)
                                delta = self.lora_B[adapter_name](self.lora_A[adapter_name](lora_input)) * self.scaling[adapter_name]
                                if self.init_noise_weights and self.adapter_noises is not None and adapter_name in self.adapter_noises:
                                    delta_noise = hidden_states @ self.adapter_noises[adapter_name].transpose(0,1)
                                    delta = delta + delta_noise

                            # Apply the activation function (if needed)
                            if self.init_activate_functions and adapter_name in self.lora_activations:
                                delta = self.lora_activations[adapter_name](delta)

                            # Check shape matching
                            if hidden_states.shape != delta.shape:
                                print("Shape mismatch detected at layer:", i, "adapter:", adapter_name)
                                print("hidden_states:", hidden_states.shape)
                                print("delta:", delta.shape)
                                # Try to align the batch dimension by truncation
                                min_bsz = min(hidden_states.size(0), delta.size(0))
                                # If hidden_states is longer than delta or delta is longer than hidden_states, truncate the larger one
                                if hidden_states.size(0) > min_bsz:
                                    hidden_states = hidden_states[:min_bsz, ...]
                                if delta.size(0) > min_bsz:
                                    delta = delta[:min_bsz, ...]
                                print("After adjustment:")
                                print("hidden_states:", hidden_states.shape)
                                print("delta:", delta.shape)

                            # Add delta to hidden_states
                            hidden_states = hidden_states + delta

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, next_decoder_cache, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

class DenseParallelEncoder(nn.Module):
    def __init__(
        self,
        base_encoder,
        lora_parallel_schedule,
        lora_dropout=0.1,
        init_lora_weights=True,
        activation_fn=nn.ReLU(),  # Default activation function
        init_activate_functions=False,
        init_noise_weights=False, # Whether to initialize random noise matrix W
        fan_in_fan_out=False,
        **kwargs
    ):
        nn.Module.__init__(self)
        # self.weight = nn.Parameter(torch.zeros(1, 1), requires_grad=False)
        self.base_encoder = base_encoder
        self.layers = base_encoder.layer
        self.in_features = self.layers[0].attention.self.query.in_features
        self.out_features = self.layers[0].attention.self.query.out_features

        # Freeze original layer parameters
        for layer in self.layers:
            for param in layer.parameters():
                param.requires_grad = False

        # Initialize LoRA parameters
        self.adapter_weights = nn.ParameterDict()
        self.lora_schedules = {}
        self.lora_activations = nn.ModuleDict() if init_activate_functions else None
        self.adapter_noises = nn.ParameterDict() if init_noise_weights else None
        self.lora_dropout = nn.ModuleDict({
            adapter_name: nn.Dropout(lora_dropout) for _, _, adapter_name in lora_parallel_schedule
        })

        for start_idx, end_idx, adapter_name in lora_parallel_schedule:
            # Initialize a single large matrix as a LoRA adapter
            adapter_weight = nn.Parameter(
                torch.zeros(self.out_features, self.in_features),
                requires_grad=True  # Ensure gradients are computed
            )
            if init_lora_weights:
                nn.init.kaiming_uniform_(adapter_weight, a=math.sqrt(5))  # Or other initialization methods

            self.adapter_weights.update({
                adapter_name: adapter_weight
            })

            # Initialize activation functions (if needed)
            if init_activate_functions:
                self.lora_activations.update({
                    adapter_name: activation_fn
                })

            # Initialize noise matrix (if needed)
            if self.adapter_noises is not None:
                noise_w = nn.Parameter(torch.zeros(self.out_features, self.in_features), requires_grad=False)
                self.adapter_noises.update({adapter_name: noise_w})

            # Define scheduling
            self.lora_schedules[adapter_name] = {
                "start_idx": start_idx,
                "end_idx": end_idx,
                "merged": False
            }

        self.disable_adapters = False
        self.input_cache = {}
        self.init_noise_weights = init_noise_weights
        self.init_activate_functions = init_activate_functions

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        # Clear input_cache
        self.input_cache.clear()

        current_device = hidden_states.device

        # Move adapter_weights and adapter_noises to the current device
        for adapter_name in self.adapter_weights.keys():
            if self.adapter_weights[adapter_name].device != current_device:
                self.adapter_weights[adapter_name].data = self.adapter_weights[adapter_name].data.to(current_device)
            if self.adapter_noises is not None and adapter_name in self.adapter_noises and self.adapter_noises[adapter_name].device != current_device:
                self.adapter_noises[adapter_name].data = self.adapter_noises[adapter_name].data.to(current_device)

        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        if self.disable_adapters:
            # Optionally disable merged adapters
            for adapter_name in self.lora_schedules:
                if self.lora_schedules[adapter_name]["merged"]:
                    self.unmerge(adapter_name)

        if past_key_values is None:
            past_key_values = [None] * len(self.layers)

        for i, layer_module in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # Cache adapter input
            if not self.disable_adapters:
                for adapter_name, sched in self.lora_schedules.items():
                    if i == sched["start_idx"]:
                        self.input_cache[adapter_name] = hidden_states
                        # print(f"Cached lora_input for adapter '{adapter_name}' at layer {i}")

            # Pass through Transformer layer
            layer_outputs = layer_module(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask[i] if head_mask is not None else None,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                past_key_value=past_key_values[i],
                output_attentions=output_attentions,
            )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[1],)

            if output_attentions:
                attn_idx = 1 if not use_cache else 2
                all_self_attentions += (layer_outputs[attn_idx],)

            # Apply adapter
            if not self.disable_adapters:
                for adapter_name, sched in self.lora_schedules.items():
                    if i == sched["end_idx"]:
                        if adapter_name in self.adapter_weights and self.adapter_weights[adapter_name].requires_grad:
                            # Compute delta
                            delta = hidden_states @ self.adapter_weights[adapter_name].transpose(0, 1)  # (batch, out_features)

                            # Add noise (if needed)
                            if self.init_noise_weights and self.adapter_noises is not None and adapter_name in self.adapter_noises:
                                delta_noise = hidden_states @ self.adapter_noises[adapter_name].transpose(0, 1)
                                delta = delta + delta_noise

                            # Apply dropout
                            delta = self.lora_dropout[adapter_name](delta)

                            # Apply activation function (if needed)
                            if self.init_activate_functions and self.lora_activations is not None and adapter_name in self.lora_activations:
                                delta = self.lora_activations[adapter_name](delta)

                            # Check shape matching
                            if hidden_states.shape != delta.shape:
                                print(f"Shape mismatch detected at layer: {i}, adapter: {adapter_name}")
                                print(f"hidden_states: {hidden_states.shape}")
                                print(f"delta: {delta.shape}")
                                # Truncate to match
                                min_bsz = min(hidden_states.size(0), delta.size(0))
                                hidden_states = hidden_states[:min_bsz, ...]
                                delta = delta[:min_bsz, ...]
                                print("After adjustment:")
                                print(f"hidden_states: {hidden_states.shape}")
                                print(f"delta: {delta.shape}")

                            # Add delta to hidden_states
                            hidden_states = hidden_states + delta

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, next_decoder_cache, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )



if is_bnb_available():

    class Linear8bitLt(bnb.nn.Linear8bitLt, LoraLayer):
        # Lora implemented in a dense layer
        def __init__(
            self,
            adapter_name,
            in_features,
            out_features,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            **kwargs,
        ):
            bnb.nn.Linear8bitLt.__init__(
                self,
                in_features,
                out_features,
                bias=kwargs.get("bias", True),
                has_fp16_weights=kwargs.get("has_fp16_weights", True),
                memory_efficient_backward=kwargs.get("memory_efficient_backward", False),
                threshold=kwargs.get("threshold", 0.0),
                index=kwargs.get("index", None),
            )
            LoraLayer.__init__(self, in_features=in_features, out_features=out_features)

            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            init_lora_weights = kwargs.pop("init_lora_weights", True)
            self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
            self.active_adapter = adapter_name

        def forward(self, x: torch.Tensor):
            result = super().forward(x)

            if self.disable_adapters or self.active_adapter not in self.lora_A.keys():
                return result
            elif self.r[self.active_adapter] > 0:
                if not torch.is_autocast_enabled():
                    expected_dtype = result.dtype

                    if x.dtype != torch.float32:
                        x = x.float()
                    output = (
                        self.lora_B[self.active_adapter](
                            self.lora_A[self.active_adapter](self.lora_dropout[self.active_adapter](x))
                        ).to(expected_dtype)
                        * self.scaling[self.active_adapter]
                    )
                else:
                    output = (
                        self.lora_B[self.active_adapter](
                            self.lora_A[self.active_adapter](self.lora_dropout[self.active_adapter](x))
                        )
                        * self.scaling[self.active_adapter]
                    )
                result += output
            return result

    if is_bnb_4bit_available():

        class Linear4bit(bnb.nn.Linear4bit, LoraLayer):
            # Lora implemented in a dense layer
            def __init__(
                self,
                adapter_name,
                in_features,
                out_features,
                r: int = 0,
                lora_alpha: int = 1,
                lora_dropout: float = 0.0,
                **kwargs,
            ):
                bnb.nn.Linear4bit.__init__(
                    self,
                    in_features,
                    out_features,
                    bias=kwargs.get("bias", True),
                    compute_dtype=kwargs.get("compute_dtype", torch.float32),
                    compress_statistics=kwargs.get("compress_statistics", True),
                    quant_type=kwargs.get("quant_type", "nf4"),
                )
                LoraLayer.__init__(self, in_features=in_features, out_features=out_features)

                # Freezing the pre-trained weight matrix
                self.weight.requires_grad = False

                init_lora_weights = kwargs.pop("init_lora_weights", True)
                self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
                self.active_adapter = adapter_name

            def forward(self, x: torch.Tensor):
                result = super().forward(x)

                if self.disable_adapters or self.active_adapter not in self.lora_A.keys():
                    return result
                elif self.r[self.active_adapter] > 0:
                    result = result.clone()
                    if not torch.is_autocast_enabled():
                        expected_dtype = result.dtype
                        x = x.to(self.lora_A[self.active_adapter].weight.dtype)
                        output = (
                            self.lora_B[self.active_adapter](
                                self.lora_A[self.active_adapter](self.lora_dropout[self.active_adapter](x))
                            ).to(expected_dtype)
                            * self.scaling[self.active_adapter]
                        )
                    else:
                        output = (
                            self.lora_B[self.active_adapter](
                                self.lora_A[self.active_adapter](self.lora_dropout[self.active_adapter](x))
                            )
                            * self.scaling[self.active_adapter]
                        )
                    result += output
                return result


class QuantLinear(torch.nn.Module, LoraLayer):
    def __init__(
        self,
        adapter_name,
        quant_linear_module,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        **kwargs,
    ):
        torch.nn.Module.__init__(self)
        LoraLayer.__init__(
            self, in_features=quant_linear_module.infeatures, out_features=quant_linear_module.outfeatures
        )
        self.quant_linear_module = quant_linear_module
        self.weight = quant_linear_module.qweight
        init_lora_weights = kwargs.pop("init_lora_weights", True)
        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
        self.active_adapter = adapter_name

    def forward(self, x: torch.Tensor):
        result = self.quant_linear_module(x)
        if self.disable_adapters or self.active_adapter not in self.lora_A.keys():
            return result
        elif self.r[self.active_adapter] > 0:
            result = result.clone()
            if not torch.is_autocast_enabled():
                expected_dtype = result.dtype
                x = x.to(self.lora_A[self.active_adapter].weight.dtype)
                output = (
                    self.lora_B[self.active_adapter](
                        self.lora_A[self.active_adapter](self.lora_dropout[self.active_adapter](x))
                    ).to(expected_dtype)
                    * self.scaling[self.active_adapter]
                )
            else:
                output = (
                    self.lora_B[self.active_adapter](
                        self.lora_A[self.active_adapter](self.lora_dropout[self.active_adapter](x))
                    )
                    * self.scaling[self.active_adapter]
                )
            result += output
        return result

    # TODO: Check if it is better as suggested by users https://github.com/PanQiWei/AutoGPTQ/pull/102
    # def reset_lora_parameters(self, adapter_name):
    #     if adapter_name in self.lora_A.keys():
    #         torch.nn.init.xavier_uniform_(self.lora_A[adapter_name].weight)
    #         torch.nn.init.zeros_(self.lora_B[adapter_name].weight)
