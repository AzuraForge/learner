# ========== DOSYA: src/azuraforge_learner/optimizers.py ==========
from typing import List
from azuraforge_core import Tensor

class Optimizer:
    def __init__(self, params: List[Tensor], lr: float):
        self.params = [p for p in params if p.requires_grad]
        self.lr = lr
    def step(self) -> None: raise NotImplementedError
    def zero_grad(self) -> None:
        for p in self.params:
            if p.grad is not None: p.grad.fill(0.0)

class SGD(Optimizer):
    def step(self) -> None:
        for p in self.params:
            if p.grad is not None: p.data -= self.lr * p.grad