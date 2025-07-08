from .base import Layer
from azuraforge_core import Tensor

class Flatten(Layer):
    """Giriş tensörünü (N, C, H, W) formatından (N, C*H*W) formatına düzleştirir."""
    def __init__(self):
        super().__init__()
        self.input_shape = None

    def forward(self, x: Tensor) -> Tensor:
        self.input_shape = x.data.shape
        N = self.input_shape[0]
        out = Tensor(x.data.reshape(N, -1), _children=(x,), _op="flatten", requires_grad=x.requires_grad)
        
        def _backward():
            if x.requires_grad and x.grad is not None:
                x.grad += out.grad.reshape(self.input_shape)
        
        out._backward = _backward
        return out
        