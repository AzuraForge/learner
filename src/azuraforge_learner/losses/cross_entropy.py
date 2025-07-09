from azuraforge_core import Tensor
from .base import Loss

class CrossEntropyLoss(Loss):
    """
    LogSoftmax ve NLLLoss'u birleştiren kayıp fonksiyonu.
    Sınıflandırma görevleri için daha verimli ve stabildir.
    Hem 2D (batch, classes) hem de 3D (batch, seq_len, classes) girdileri destekler.
    """
    def __call__(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        # 1. Adım: Logit'leri log olasılıklarına çevir.
        log_probs = y_pred.log_softmax(axis=-1)
        
        # 2. Adım: NLLLoss'u hesapla.
        # Bu metot artık 3D tensörleri ve gradyanları doğru bir şekilde işliyor.
        return log_probs.nll_loss(y_true)