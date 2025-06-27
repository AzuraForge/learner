# ========== GÜNCELLENMİŞ DOSYA: src/azuraforge_learner/callbacks.py ==========
import logging
import os
import numpy as np
from .events import Event

class Callback:
    """
    Tüm callback'lerin temel sınıfı. Olayları dinler ve ilgili metoda yönlendirir.
    """
    def __call__(self, event: Event):
        # Gelen olayın adına göre doğru metodu çağır (örn: on_epoch_end)
        method = getattr(self, f"on_{event.name}", None)
        if method:
            method(event)

    def on_train_begin(self, event: Event): pass
    def on_train_end(self, event: Event): pass
    def on_epoch_begin(self, event: Event): pass
    def on_epoch_end(self, event: Event): pass
    def on_batch_begin(self, event: Event): pass
    def on_batch_end(self, event: Event): pass

class ModelCheckpoint(Callback):
    def __init__(self, filepath: str, monitor: str = "val_loss", mode: str = "min", verbose: int = 1):
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.verbose = verbose
        self.best = np.inf if mode == "min" else -np.inf

    def on_epoch_end(self, event: Event):
        current_val = event.payload.get(self.monitor)
        if current_val is None:
            return

        is_better = (self.mode == "min" and current_val < self.best) or \
                    (self.mode == "max" and current_val > self.best)

        if is_better:
            if self.verbose > 0:
                print(f"ModelCheckpoint: {self.monitor} improved from {self.best:.6f} to {current_val:.6f}. Saving model to {self.filepath}")
            self.best = current_val
            # Learner'a modeli kaydetmesini söylüyoruz (doğrudan kendimiz kaydetmiyoruz)
            event.learner.save(self.filepath)

class EarlyStopping(Callback):
    def __init__(self, monitor: str = "val_loss", patience: int = 10, mode: str = "min"):
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.wait = 0
        self.best = np.inf if mode == "min" else -np.inf

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
                print(f"EarlyStopping: Stopping training. {self.monitor} did not improve for {self.patience} epochs.")
                event.learner.stop_training = True