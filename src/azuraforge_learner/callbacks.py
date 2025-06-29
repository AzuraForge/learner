import os
import numpy as np
from .events import Event

# 'Learner' sınıfı henüz tanımlanmadığı için ileriye dönük referans (forward reference) kullanıyoruz
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .learner import Learner

class Callback:
    """
    Tüm callback'lerin temel sınıfı. Olayları dinler ve ilgili metoda yönlendirir.
    """
    def __call__(self, event: Event):
        method = getattr(self, f"on_{event.name}", None)
        if method:
            method(event)

    def on_train_begin(self, event: Event): pass
    def on_train_end(self, event: Event): pass
    def on_epoch_begin(self, event: Event): pass
    def on_epoch_end(self, event: Event): pass

class ModelCheckpoint(Callback):
    """Her epoch sonunda performansı izler ve sadece en iyi modeli kaydeder."""
    def __init__(self, filepath: str, monitor: str = "val_loss", mode: str = "min", verbose: int = 1):
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.verbose = verbose
        self.best = np.inf if mode == "min" else -np.inf

        # Dosyanın kaydedileceği dizini oluştur
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
                print(f"ModelCheckpoint: {self.monitor} improved from {self.best:.6f} to {current_val:.6f}. Saving model to {self.filepath}")
            self.best = current_val
            event.learner.save(self.filepath)

class EarlyStopping(Callback):
    """Performans belirli bir epoch sayısı boyunca iyileşmediğinde eğitimi durdurur."""
    def __init__(self, monitor: str = "val_loss", patience: int = 10, mode: str = "min", verbose: int = 1):
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.verbose = verbose
        self.wait = 0
        self.best = np.inf if mode == "min" else -np.inf

    def on_train_begin(self, event: Event):
        # Eğitimin başında sayaçları sıfırla
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
                event.learner.stop_training = True
