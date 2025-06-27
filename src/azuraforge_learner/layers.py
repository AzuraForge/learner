# ========== DOSYA: src/azuraforge_learner/layers.py ==========
from typing import List
import numpy as np
from azuraforge_core import Tensor, xp

class Layer:
    def forward(self, *args, **kwargs) -> Tensor: raise NotImplementedError
    def parameters(self) -> List[Tensor]: return []
    def __call__(self, *args, **kwargs) -> Tensor: return self.forward(*args, **kwargs)

class Linear(Layer):
    def __init__(self, input_features: int, output_features: int, bias: bool = True):
        limit = np.sqrt(2.0 / input_features) if input_features > 0 else 0.1
        self.weights = Tensor(xp.random.randn(input_features, output_features) * limit, requires_grad=True)
        self.bias = Tensor(xp.zeros(output_features), requires_grad=True) if bias else None
    def forward(self, x: Tensor) -> Tensor:
        out = x.dot(self.weights)
        return out + self.bias if self.bias is not None else out
    def parameters(self) -> List[Tensor]:
        return [self.weights] + ([self.bias] if self.bias is not None else [])