# ========== DOSYA: src/azuraforge_learner/callbacks.py ==========
import numpy as np
from .events import Event

class Callback:
    def __call__(self, event: Event):
        method = getattr(self, f"on_{event.name}", None)
        if method:
            method(event)
    def on_train_begin(self, event: Event): pass
    def on_train_end(self, event: Event): pass
    def on_epoch_begin(self, event: Event): pass
    def on_epoch_end(self, event: Event): pass

class ModelCheckpoint(Callback):
    def __init__(self, filepath: str, monitor: str = "val_loss", mode: str = "min"):
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.best = np.inf if mode == "min" else -np.inf

    def on_epoch_end(self, event: Event):
        current_val = event.payload.get(self.monitor)
        if current_val is None: return
        if (self.mode == "min" and current_val < self.best) or \
           (self.mode == "max" and current_val > self.best):
            self.best = current_val
            print(f"ModelCheckpoint: {self.monitor} improved. Saving model to {self.filepath}")
            event.learner.save(self.filepath)