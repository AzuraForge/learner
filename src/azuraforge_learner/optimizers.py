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

# YENİ: Adam Optimizer eklendi
class Adam(Optimizer):
    def __init__(self, params: List[Tensor], lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
        super().__init__(params, lr)
        from azuraforge_core import xp # xp'yi burada import ediyoruz
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {id(p): xp.zeros_like(p.data) for p in self.params}
        self.v = {id(p): xp.zeros_like(p.data) for p in self.params}
        self.t = 0

    def step(self) -> None:
        self.t += 1
        from azuraforge_core import xp # xp'yi burada import ediyoruz
        for p in self.params:
            if p.grad is not None:
                param_id = id(p)
                self.m[param_id] = self.beta1 * self.m[param_id] + (1 - self.beta1) * p.grad
                self.v[param_id] = self.beta2 * self.v[param_id] + (1 - self.beta2) * (p.grad**2)

                m_hat = self.m[param_id] / (1 - self.beta1**self.t)
                v_hat = self.v[param_id] / (1 - self.beta2**self.t)

                update_val = self.lr * m_hat / (xp.sqrt(v_hat) + self.epsilon)
                p.data -= update_val