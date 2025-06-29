# learner/src/azuraforge_learner/learner.py

import time
from typing import Any, Dict, List, Optional
import numpy as np

from azuraforge_core import Tensor
from .events import Event
from .models import Sequential
from .losses import Loss
from .optimizers import Optimizer
from .callbacks import Callback

class Learner:
    def __init__(self, model: Sequential, criterion: Loss, optimizer: Optimizer, callbacks: Optional[List[Callback]] = None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.callbacks = callbacks or []
        # Tüm callback'lere learner referansını ata
        for cb in self.callbacks:
            if hasattr(cb, 'set_learner'): # Geriye dönük uyumluluk için
                 cb.set_learner(self)
                 
        self.history: Dict[str, List[float]] = {}
        self.stop_training: bool = False

    def _publish(self, event_name: str, payload: Optional[Dict[str, Any]] = None):
        """
        Olayı, kendisine atanmış tüm callback'lere yayınlar.
        Artık Redis veya başka bir teknoloji hakkında hiçbir şey bilmiyor.
        """
        event = Event(name=event_name, learner=self, payload=payload or {})
        for cb in self.callbacks:
            # Callback'in ilgili olay metodunu (örn: on_epoch_end) çağırır.
            cb(event)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, epochs: int, pipeline_name: str = "Bilinmiyor"):
        self.history = {"loss": []}
        X_train_t, y_train_t = Tensor(X_train), Tensor(y_train)
        
        self._publish("train_begin", payload={"total_epochs": epochs, "status_text": "Eğitim başlıyor...", "pipeline_name": pipeline_name})
        
        for epoch in range(epochs):
            if self.stop_training:
                break
            
            self._publish("epoch_begin", payload={"epoch": epoch, "total_epochs": epochs, "pipeline_name": pipeline_name})
            
            y_pred = self.model(X_train_t)
            loss = self.criterion(y_pred, y_train_t)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            current_loss = loss.to_cpu().item() if hasattr(loss, 'to_cpu') else float(loss.data)
            
            epoch_logs = {
                "epoch": epoch + 1,
                "total_epochs": epochs,
                "loss": current_loss,
                "status_text": f"Epoch {epoch + 1}/{epochs} tamamlandı, Kayıp: {current_loss:.6f}",
                "pipeline_name": pipeline_name
            }
            
            self.history["loss"].append(current_loss)
            self._publish("epoch_end", payload=epoch_logs)
            
        self._publish("train_end", payload={"status_text": "Eğitim tamamlandı.", "pipeline_name": pipeline_name})
        return self.history