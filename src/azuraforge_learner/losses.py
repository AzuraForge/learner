# ========== DOSYA: src/azuraforge_learner/losses.py ==========
from azuraforge_core import Tensor, _ensure_tensor
import numpy as np

class Loss:
    def __call__(self, y_pred: Tensor, y_true: Tensor) -> Tensor: raise NotImplementedError

class MSELoss(Loss):
    def __call__(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        return ((_ensure_tensor(y_pred) - _ensure_tensor(y_true)) ** 2).mean()

class BCELoss(Loss):
    def __init__(self, epsilon: float = 1e-12): self.epsilon = epsilon
    def __call__(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        y_pred = _ensure_tensor(y_pred).clip(self.epsilon, 1.0 - self.epsilon)
        y_true = _ensure_tensor(y_true)
        return -(y_true * y_pred.log() + (1 - y_true) * (1 - y_pred).log()).mean()

class CategoricalCrossentropyLoss(Loss):
    # ... (CCE Loss içeriği eski projeden alınabilir)
    pass