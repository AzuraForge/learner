import numpy as np
from typing import List
from azuraforge_core import Tensor, xp
from .base import Layer

class Conv2D(Layer):
    """2D Konvolüsyon katmanı."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # He/Kaiming başlatması
        limit = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.weights = Tensor(xp.random.randn(out_channels, in_channels, kernel_size, kernel_size) * limit, requires_grad=True)
        self.bias = Tensor(xp.zeros(out_channels), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        return x.conv2d(self.weights, self.bias, self.stride, self.padding)

    def parameters(self) -> List[Tensor]:
        return [self.weights, self.bias]

class MaxPool2D(Layer):
    """2D Max Pooling katmanı."""
    def __init__(self, kernel_size: int = 2, stride: int = 2):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        return x.max_pool2d(self.kernel_size, self.stride)
        