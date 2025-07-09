from azuraforge_core import Tensor
from .base import Loss

class MSELoss(Loss):
    """
    Mean Squared Error (Ortalama Karesel Hata) kayıp fonksiyonu.
    Genellikle regresyon görevlerinde kullanılır.
    """
    def __call__(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        return ((y_pred - y_true) ** 2).mean()

