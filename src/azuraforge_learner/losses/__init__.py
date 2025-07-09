# Bu dosya, tüm kayıp fonksiyonlarını tek bir yerden kolayca import etmeyi sağlar.
from .base import Loss
from .mse import MSELoss
from .cross_entropy import CrossEntropyLoss

__all__ = ["Loss", "MSELoss", "CrossEntropyLoss"]

