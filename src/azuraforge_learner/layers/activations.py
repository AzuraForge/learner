from .base import Layer
from azuraforge_core import Tensor

class ReLU(Layer):
    """Rectified Linear Unit aktivasyon katmanı."""
    def forward(self, x: Tensor) -> Tensor:
        return x.relu()

class Sigmoid(Layer):
    """Sigmoid aktivasyon katmanı."""
    def forward(self, x: Tensor) -> Tensor:
        return x.sigmoid()
        
# Tanh gibi diğer aktivasyonlar da buraya eklenebilir.