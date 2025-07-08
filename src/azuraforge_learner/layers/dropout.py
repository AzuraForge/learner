from azuraforge_core import Tensor, xp
from .base import Layer

class Dropout(Layer):
    """
    Eğitim sırasında aşırı öğrenmeyi (overfitting) önlemek için kullanılan Dropout katmanı.
    Tahmin (eval) modunda hiçbir şey yapmaz.
    """
    def __init__(self, p: float = 0.5):
        super().__init__()
        if not (0.0 <= p < 1.0):
            raise ValueError(f"Dropout olasılığı 0 ile 1 arasında olmalı, ancak {p} girildi.")
        self.p = p
        self.is_training = True

    def forward(self, x: Tensor) -> Tensor:
        if not self.is_training:
            return x
        
        # Inverted Dropout: Ölçeklemeyi eğitim sırasında yapıyoruz.
        mask = (xp.random.rand(*x.data.shape) > self.p) / (1.0 - self.p)
        out_data = x.data * mask
        
        out = Tensor(out_data, _children=(x,), _op="dropout", requires_grad=x.requires_grad)

        def _backward():
            if x.requires_grad and x.grad is not None:
                x.grad += out.grad * mask

        out._backward = _backward
        return out

    def eval(self):
        """Modeli tahmin/değerlendirme moduna alır, Dropout'u devre dışı bırakır."""
        self.is_training = False

    def train(self):
        """Modeli eğitim moduna alır, Dropout'u etkinleştirir."""
        self.is_training = True
