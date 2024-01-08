import json
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
from sglang.srt.managers.router.infer_batch import ForwardMode
from torch import nn
from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.weight_utils import (
    default_weight_loader,
    hf_model_weights_iterator,
)


class LoRAModel(nn.Module):
    def __init__(
        self,
        config,
    ) -> None:
        self.config = config

    def forward(
        self,
    ) -> torch.Tensor:
        pass

    def load_weights(
        self,
        device="cpu",
    ):
        pass


class LoRABatch(nn.Module):
    def __init__() -> None:
        pass
