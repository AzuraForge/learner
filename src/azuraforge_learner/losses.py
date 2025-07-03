from azuraforge_core import Tensor

class Loss:
    def __call__(self, y_pred: Tensor, y_true: Tensor) -> Tensor: raise NotImplementedError

class MSELoss(Loss):
    def __call__(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        return ((y_pred - y_true) ** 2).mean()

class CrossEntropyLoss(Loss):
    """
    LogSoftmax ve NLLLoss'u birleştiren kayıp fonksiyonu.
    Sınıflandırma görevleri için daha verimli ve stabildir.
    """
    def __call__(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        # y_pred (logits) -> log_softmax -> nll_loss
        log_probs = y_pred.log_softmax(axis=-1)
        return log_probs.nll_loss(y_true)