# learner/src/azuraforge_learner/callbacks.py

import os
import numpy as np
from typing import TYPE_CHECKING, Optional, Any
from .events import Event # Event'i de import edelim

# Döngüsel importu önlemek için, sadece tip kontrolü sırasında Learner'ı import et
if TYPE_CHECKING:
    from .learner import Learner

class Callback:
    """
    Tüm callback'lerin temel sınıfı.
    Kendisini çalıştıran Learner'a bir referans tutar.
    """
    def __init__(self):
        self.learner: Optional['Learner'] = None

    def set_learner(self, learner: 'Learner'):
        """Bu metod, Learner tarafından çağrılarak referansı ayarlar."""
        self.learner = learner

    def __call__(self, event: Event):
        """
        Gelen olaya göre ilgili metodu (örn: on_epoch_end) çağırır.
        """
        method = getattr(self, f"on_{event.name}", None)
        if method:
            method(event)

    # Olay metotları
    def on_train_begin(self, event: Event) -> None: pass
    def on_train_end(self, event: Event) -> None: pass
    def on_epoch_begin(self, event: Event) -> None: pass
    def on_epoch_end(self, event: Event) -> None: pass
    def on_batch_begin(self, event: Event) -> None: pass
    def on_batch_end(self, event: Event) -> None: pass


# ÖNEMLİ: ModelCheckpoint ve EarlyStopping sınıflarını koruyoruz ve
# yeni temel sınıftan miras almalarını sağlıyoruz.
class ModelCheckpoint(Callback):
    """Her epoch sonunda performansı izler ve sadece en iyi modeli kaydeder."""
    def __init__(self, filepath: str, monitor: str = "val_loss", mode: str = "min", verbose: int = 1):
        super().__init__() # Temel sınıfın init'ini çağır
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.verbose = verbose
        self.best = np.inf if mode == "min" else -np.inf

        dir_path = os.path.dirname(self.filepath)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

    def on_epoch_end(self, event: Event):
        current_val = event.payload.get(self.monitor)
        if current_val is None:
            if event.payload.get("epoch") == 0 and self.verbose > 0:
                print(f"ModelCheckpoint Warning: Can't find metric '{self.monitor}' to save model.")
            return

        is_better = (self.mode == "min" and current_val < self.best) or \
                    (self.mode == "max" and current_val > self.best)

        if is_better:
            if self.verbose > 0:
                print(f"ModelCheckpoint: {self.monitor} improved from {self.best:.6f} to {current_val:.6f}. Saving model...")
            self.best = current_val
            if self.learner and hasattr(self.learner, 'save_model'): # Learner'da save_model metodu varsa
                 self.learner.save_model(self.filepath)


class EarlyStopping(Callback):
    """Performans belirli bir epoch sayısı boyunca iyileşmediğinde eğitimi durdurur."""
    def __init__(self, monitor: str = "val_loss", patience: int = 10, mode: str = "min", verbose: int = 1):
        super().__init__() # Temel sınıfın init'ini çağır
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.verbose = verbose
        self.wait = 0
        self.best = np.inf if mode == "min" else -np.inf

    def on_train_begin(self, event: Event):
        self.wait = 0
        self.best = np.inf if self.mode == "min" else -np.inf

    def on_epoch_end(self, event: Event):
        current_val = event.payload.get(self.monitor)
        if current_val is None:
            return
            
        is_better = (self.mode == "min" and current_val < self.best) or \
                    (self.mode == "max" and current_val > self.best)

        if is_better:
            self.best = current_val
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                if self.verbose > 0:
                    print(f"EarlyStopping: Stopping training. {self.monitor} did not improve for {self.patience} epochs.")
                if self.learner:
                    self.learner.stop_training = True