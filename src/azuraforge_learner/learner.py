# ========== ÖRNEK DOSYA: src/azuraforge_learner/layers.py ==========
from typing import List
# Artık kendi motorumuzdan Tensor'u import ediyoruz!
from azuraforge_core import Tensor 

class Layer:
    def forward(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError
    def parameters(self) -> List[Tensor]:
        return []
    def __call__(self, *args, **kwargs) -> Tensor:
        return self.forward(*args, **kwargs)

class Linear(Layer):
    def __init__(self, input_features: int, output_features: int):
        # Ağırlıklarımızı Tensor nesneleri olarak oluşturuyoruz
        self.weights = Tensor(..., requires_grad=True)
        self.bias = Tensor(..., requires_grad=True)
    
    def forward(self, x: Tensor) -> Tensor:
        return x.dot(self.weights) + self.bias
        
    def parameters(self) -> List[Tensor]:
        return [self.weights, self.bias]