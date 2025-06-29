from typing import List
import numpy as np
from azuraforge_core import Tensor, xp

class Layer:
    def forward(self, x: Tensor) -> Tensor: raise NotImplementedError
    def parameters(self) -> List[Tensor]: return []
    def __call__(self, x: Tensor) -> Tensor: return self.forward(x)

class Linear(Layer):
    def __init__(self, input_dim: int, output_dim: int):
        limit = np.sqrt(2.0 / input_dim)
        self.weights = Tensor(xp.random.randn(input_dim, output_dim) * limit, requires_grad=True)
        self.bias = Tensor(xp.zeros(output_dim), requires_grad=True)
    def forward(self, x: Tensor) -> Tensor:
        return x.dot(self.weights) + self.bias
    def parameters(self) -> List[Tensor]:
        return [self.weights, self.bias]

class ReLU(Layer):
    def forward(self, x: Tensor) -> Tensor:
        return x.relu()
