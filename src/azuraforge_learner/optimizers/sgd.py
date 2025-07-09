from .base import Optimizer

class SGD(Optimizer):
    def step(self) -> None:
        for p in self.params:
            if p.grad is not None:
                p.data -= self.lr * p.grad