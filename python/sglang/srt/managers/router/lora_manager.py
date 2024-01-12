from sglang.srt.models.lora import LoRAModel
from sglang.srt.model_config import LoRAConfig


class LoRAManager:
    def __init__(
        self,
        base_model,
        lora_paths,
        base_config,
        device="cpu",
    ):
        self.base_model = base_model
        self.lora_paths = lora_paths
        self.base_config = base_config

        self.init_loras(device=device)

    def init_loras(self, device):
        pass
        # # get modules that need to be monkey patched from the base model (qkv_proj, o_proj)
        # modules = get_lora_modules()  # (self_attn, mlp)

        # for module in modules:
        #     # monkey patch to use Lora version
        #     new_module = apply_lora(module)
        #     setattr(module.parent, module.name) = new_module

        #     # load all weights to CPU
        #     lora_weights = []
        #     for lora_path in self.lora_paths:
        #         lora_weight_name = ...
        #         lora_weight = load_weights(lora_weight_name)
        #         lora_weights.append(lora_weight)
        #     new_moduel.lora_weights = lora_weights

        # # read the corres

        # get_

        # self.configs = {}
        # self.loras = {}
        # for path in self.lora_paths:
        #     self.configs[path] = LoRAConfig(path)
        #     self.loras[path] = LoRAModel(self.configs[path], self.base_config)
        #     self.loras[path].load_weights(path, device)

    def add_loras(self, lora_paths):
        pass
        # for module in self.modules:
        #     module.load_to_gpu(lora_ids)
