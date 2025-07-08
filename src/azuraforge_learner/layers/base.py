from typing import List
from azuraforge_core import Tensor

class Layer:
    """Tüm katmanların miras alacağı soyut temel sınıf."""
    def forward(self, x: Tensor) -> Tensor:
        """Katmanın ileri geçiş (forward pass) mantığını tanımlar."""
        raise NotImplementedError

    def parameters(self) -> List[Tensor]:
        """Katmanın eğitilebilir parametrelerini bir liste olarak döndürür."""
        return []

    def __call__(self, x: Tensor) -> Tensor:
        """Katmanın bir fonksiyon gibi çağrılabilmesini sağlar (örn: layer(x))."""
        return self.forward(x)
