import numpy as np
from typing import List
from azuraforge_core import Tensor, xp
from .base import Layer

class Linear(Layer):
    """Tam bağlantılı (fully-connected) doğrusal katman."""
    def __init__(self, input_dim: int, output_dim: int):
        # He/Kaiming başlatması (ReLU ile iyi çalışır)
        limit = np.sqrt(2.0 / input_dim)
        self.weights = Tensor(xp.random.randn(input_dim, output_dim) * limit, requires_grad=True)
        self.bias = Tensor(xp.zeros(output_dim), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        return x.dot(self.weights) + self.bias

    def parameters(self) -> List[Tensor]:
        return [self.weights, self.bias]
