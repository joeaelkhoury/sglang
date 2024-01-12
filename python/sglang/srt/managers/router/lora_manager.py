from sglang.srt.model_config import LoRAConfig
from sglang.srt.models.lora import LoRAAttention, LoRAMLP


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

    def create_lora_module(self, module_name, module, lora_path):
        if "self_attn" in module_name:
            return LoRAAttention()
        elif "mlp" in module_name:
            return LoRAMLP()
        # TODO
        # should we rewrite the whole attention and mlp?
        # disable radixattention

    def init_loras(self, device):
        self.configs = {}
        for path in self.lora_paths:
            self.configs[path] = LoRAConfig(path)
            # get modules that need to be monkey patched from the base model
            modules = self.base_model.get_lora_modules(self.configs[path])
            for parent, module_name, module in modules:
                # monkey patch to use Lora version
                new_module = self.create_lora_module(module_name, module, path)
            #     setattr(parent, module_name) = new_module

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
        pass

    def add_loras(self, lora_paths):
        pass
        # for module in self.modules:
        #     module.load_to_gpu(lora_ids)
