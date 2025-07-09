from typing import List
from azuraforge_core import Tensor, xp

class Optimizer:
    def __init__(self, params: List[Tensor], lr: float):
        self.params = [p for p in params if p.requires_grad]
        self.lr = lr
    
    def step(self) -> None: 
        raise NotImplementedError
    
    def zero_grad(self) -> None:
        for p in self.params:
            if p.grad is not None:
                p.grad.fill(0.0)

    def clip_gradients(self, max_norm: float):
        total_norm = 0.0
        for p in self.params:
            if p.grad is not None:
                param_norm = xp.sum(p.grad**2)
                total_norm += param_norm
        
        total_norm = xp.sqrt(total_norm)
        
        if total_norm > max_norm:
            clip_coef = max_norm / (total_norm + 1e-6)
            for p in self.params:
                if p.grad is not None:
                    p.grad *= clip_coef