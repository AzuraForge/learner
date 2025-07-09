from azuraforge_core import Tensor, xp
from .base import Loss

class CrossEntropyLoss(Loss):
    """
    LogSoftmax ve NLLLoss'u birleştiren kayıp fonksiyonu.
    Sınıflandırma görevleri için daha verimli ve stabildir.
    Hem 2D (batch, classes) hem de 3D (batch, seq_len, classes) girdileri destekler.
    """
    def __call__(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        num_dims = y_pred.data.ndim
        
        if num_dims == 3:
            N, T, C = y_pred.data.shape
            logits_reshaped = y_pred.data.reshape(N * T, C)
            targets_reshaped = y_true.data.reshape(N * T)
        elif num_dims == 2:
            logits_reshaped = y_pred.data
            targets_reshaped = y_true.data
        else:
            raise ValueError(f"CrossEntropyLoss, 2D veya 3D tahmin tensörleri bekler, ancak {num_dims}D मिला.")

        # Sayısal stabilite için Log-Sum-Exp hilesiyle LogSoftmax
        max_logits = xp.max(logits_reshaped, axis=1, keepdims=True)
        stable_logits = logits_reshaped - max_logits
        log_sum_exp = xp.log(xp.sum(xp.exp(stable_logits), axis=1, keepdims=True))
        log_probs = stable_logits - log_sum_exp
        
        # NLLLoss hesaplaması
        num_total_samples = targets_reshaped.shape[0]
        correct_log_probs = log_probs[range(num_total_samples), targets_reshaped.astype(xp.int32)]
        
        # Ortalama kayıp
        loss_value = -xp.sum(correct_log_probs) / num_total_samples
        
        # Geri yayılım için gradyanı hesapla
        # Önce olasılıkları (softmax) bul
        probs = xp.exp(log_probs)
        # Doğru etiketlerin olasılığından 1 çıkar
        probs[range(num_total_samples), targets_reshaped.astype(xp.int32)] -= 1
        # Gradyanı örnek sayısına böl
        grad = probs / num_total_samples

        def _backward():
            if y_pred.requires_grad and y_pred.grad is not None:
                # Gradyanı orijinal şekle geri getir
                y_pred.grad += grad.reshape(y_pred.data.shape)

        # Geriye yayılım için gerekli bilgileri içeren yeni bir Tensor oluştur
        out = Tensor(loss_value, _children=(y_pred,), _op='CrossEntropyLoss')
        out._backward = _backward

        return out

