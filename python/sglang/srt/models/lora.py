import json
import os
from typing import Any, Dict, List, Optional, Tuple

import safetensors.torch
import torch
from sglang.srt.managers.router.infer_batch import ForwardMode
from sglang.srt.managers.router.model_runner import InputMetadata
from torch import nn
from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.weight_utils import (
    default_weight_loader,
    hf_model_weights_iterator,
)


class LoRAAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # self.lora_weights_cpu

    def load_adapters_to_gpu(self, lora_ids, mem_pool):
        # move self.lora_weights_cpu[lora_ids] to mem_pool
        # return address
        pass

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # get lora related things from input_metadata
        # apply lora

        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, input_metadata)
        output, _ = self.o_proj(attn_output)
        return output




class LoRAMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, input_metadata):

        # get lora related things from input_metadata
        # apply lora

        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x
