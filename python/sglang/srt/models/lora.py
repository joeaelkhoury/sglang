import json
import os
from typing import Any, Dict, List, Optional, Tuple

import safetensors.torch
import torch
from sglang.srt.managers.router.infer_batch import ForwardMode
from torch import nn
from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.weight_utils import (
    default_weight_loader,
    hf_model_weights_iterator,
)


class LoRAAttention(nn.Module):
    def __init__(
        self,
    ) -> None:
        super().__init__()

        for name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            setattr(self, name, LoRALinear())


class LoRAMLP(nn.Module):
    def __init__(
        self,
    ) -> None:
        super().__init__()


class LoRALayer(nn.Module):
    def __init__(
        self,
        layer_id,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.self_attn = LoRAAttention()
        self.mlp = LoRAMLP()
        # self.q_lora_A = None
        # self.q_lora_B = None
        # self.k_lora_A = None
        # self.k_lora_B = None
        # self.v_lora_A = None
        # self.v_lora_B = None

        # self.down_proj_lora_A = None
        # self.down_proj_lora_B = None
        # self.up_proj_lora_A = None
        # self.up_proj_lora_B = None
        # self.gate_proj_lora_A = None
        # self.gate_proj_lora_B = None


class LoRAModel(nn.Module):
    def __init__(
        self,
        config,
        base_config,
    ) -> None:
        super().__init__()
        self.config = config
        self.base_config = base_config

        self.input_embed = None
        self.output_embed = None
        self.lm_head_lora_embed_A = None
        self.lm_head_lora_embed_B = None

        self.layers = nn.ModuleList([
            LoRALayer(i)
            for i in range(base_config.num_hidden_layers)
        ])

    def load_weights(
        self,
        model_name_or_path: str,
        device: str,
        cache_dir: Optional[str] = None,
        load_format: str = "auto",
        revision: Optional[str] = None,
 
    ):
        params_dict = dict(self.named_parameters())
        print(params_dict)
        # for name, loaded_weight in hf_model_weights_iterator(
        #     model_name_or_path, cache_dir, load_format, revision
        # ):
        #     param = params_dict[name]
        #     weight_loader = getattr(param, "weight_loader", default_weight_loader)
        #     weight_loader(param, loaded_weight)
        #     loaded_weight.to(device)
        pass


class LoRABatch:
    def __init__() -> None:
        pass
