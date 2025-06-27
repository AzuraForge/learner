# ========== DOSYA: src/azuraforge_learner/models.py ==========
from typing import List, Sequence
from .layers import Layer
from azuraforge_core import Tensor

class Sequential(Layer):
    def __init__(self, *layers: Layer):
        self.layers = layers
    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
    def parameters(self) -> List[Tensor]:
        return [p for layer in self.layers for p in layer.parameters()]