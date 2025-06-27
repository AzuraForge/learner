# ========== DOSYA: src/azuraforge_learner/losses.py ==========
from azuraforge_core import Tensor
class Loss:
    def __call__(self, y_pred: Tensor, y_true: Tensor) -> Tensor: raise NotImplementedError

class MSELoss(Loss):
    def __call__(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        return ((y_pred - y_true) ** 2).mean()