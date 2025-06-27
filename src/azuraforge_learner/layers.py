# ========== DOSYA: src/azuraforge_learner/layers.py ==========
from typing import List, Tuple as TypingTuple, Optional, Any
import numpy as np
from azuraforge_core import Tensor, xp, ArrayType

class Layer:
    def forward(self, *args: Any, **kwargs: Any) -> Tensor: raise NotImplementedError
    def parameters(self) -> List[Tensor]: return []
    def __call__(self, *args: Any, **kwargs: Any) -> Tensor: return self.forward(*args, **kwargs)

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

class ReLU(Layer):
    def forward(self, x: Tensor) -> Tensor: return x.relu()

class Sigmoid(Layer):
    def forward(self, x: Tensor) -> Tensor: return x.sigmoid()

class Tanh(Layer):
    def forward(self, x: Tensor) -> Tensor: return x.tanh()
    
class Softmax(Layer):
    def __init__(self, axis: int = -1): self.axis = axis
    def forward(self, x: Tensor) -> Tensor: return x.softmax(axis=self.axis)

class LSTM(Layer):
    # Bu sınıfın tam içeriği çok uzun olduğu için şimdilik iskeletini koyuyorum.
    # Eski projeden tam içeriği buraya kopyalayabilirsin.
    def __init__(self, input_size: int, hidden_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size
        # ... Ağırlıkların başlatılması ...
    def forward(self, x: Tensor) -> Tensor:
        # ... LSTM ileri yayılım mantığı ...
        raise NotImplementedError("LSTM.forward not fully implemented in this snippet.")
    def parameters(self) -> List[Tensor]:
        # return [self.W_x, self.W_h, self.b]
        raise NotImplementedError("LSTM.parameters not fully implemented in this snippet.")