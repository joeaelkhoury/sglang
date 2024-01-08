from sglang.srt.models.lora import LoRAModel
from sglang.srt.model_config import LoRAConfig


class LoRAManager:
    def __init__(
        self,
        lora_paths,
        device="cpu",
    ):
        self.lora_paths = lora_paths

        self.init_loras(lora_paths, device=device)

    def init_loras(self, paths, device):
        self.configs = {}
        self.loras = {}
        for path in paths:
            self.configs[path] = LoRAConfig(path)
            self.loras[path] = LoRAModel(self.configs[path])
            self.loras[path].load_weights(device=device)

    def add_loras(self, lora_paths):
        pass
