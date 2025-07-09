from azuraforge_core import Tensor

class Loss:
    """Tüm kayıp fonksiyonlarının miras alacağı soyut temel sınıf."""
    def __call__(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """Kayıp fonksiyonunun ileri geçiş (forward pass) mantığını tanımlar."""
        raise NotImplementedError

