from .base import Optimizer
from azuraforge_core import Tensor, xp
from typing import List

class Adam(Optimizer):
    def __init__(self, params: List[Tensor], lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
        super().__init__(params, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {id(p): xp.zeros_like(p.data) for p in self.params}
        self.v = {id(p): xp.zeros_like(p.data) for p in self.params}
        self.t = 0

    def step(self) -> None:
        self.t += 1
        for p in self.params:
            if p.grad is not None:
                param_id = id(p)
                self.m[param_id] = self.beta1 * self.m[param_id] + (1 - self.beta1) * p.grad
                self.v[param_id] = self.beta2 * self.v[param_id] + (1 - self.beta2) * (p.grad**2)

                m_hat = self.m[param_id] / (1 - self.beta1**self.t)
                v_hat = self.v[param_id] / (1 - self.beta2**self.t)

                update_val = self.lr * m_hat / (xp.sqrt(v_hat) + self.epsilon)
                p.data -= update_val