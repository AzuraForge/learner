import os
import json
from typing import List
from .layers import Layer, Dropout # Dropout'u import edelim
from azuraforge_core import Tensor, xp

class Sequential(Layer):
    def __init__(self, *layers: Layer):
        self.layers = list(layers)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def parameters(self) -> List[Tensor]:
        return [p for layer in self.layers for p in layer.parameters()]
        
    def save(self, filepath: str):
        # ... (save metodu aynı kalıyor) ...
        dir_path = os.path.dirname(filepath)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        with open(filepath, 'w') as f:
            weights_to_save = [p.to_cpu().tolist() for p in self.parameters()]
            json.dump(weights_to_save, f)
        print(f"Model parameters saved to {filepath}")

    def load(self, filepath: str):
        # ... (load metodu aynı kalıyor) ...
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found at {filepath}")
        with open(filepath, 'r') as f:
            loaded_weights = json.load(f)
        params = self.parameters()
        if len(loaded_weights) != len(params):
            raise ValueError("Loaded weights count does not match model parameters count.")
        for i, weight_data in enumerate(loaded_weights):
            params[i].data = xp.array(weight_data, dtype=xp.float32)
        print(f"Model parameters loaded from {filepath}")

    # === YENİ METOTLAR ===
    def eval(self):
        """Modeli ve tüm Dropout katmanlarını değerlendirme moduna alır."""
        for layer in self.layers:
            if isinstance(layer, Dropout):
                layer.eval()

    def train(self):
        """Modeli ve tüm Dropout katmanlarını eğitim moduna alır."""
        for layer in self.layers:
            if isinstance(layer, Dropout):
                layer.train()
