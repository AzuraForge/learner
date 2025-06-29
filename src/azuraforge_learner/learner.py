import pickle
import time  # YENİ: time modülünü import ediyoruz
from typing import Any, Dict, List, Optional
import numpy as np
from azuraforge_core import Tensor
from .events import Event
from .models import Sequential
from .losses import Loss
from .optimizers import Optimizer
from .callbacks import Callback

class Learner:
    def __init__(self, model: Sequential, criterion: Loss, optimizer: Optimizer, callbacks: Optional[List[Callback]] = None, current_task: Optional[Any] = None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.callbacks = callbacks or []
        self.history: Dict[str, List[float]] = {}
        self.stop_training: bool = False
        self.current_task = current_task

    def _publish(self, event_name: str, payload: Optional[Dict[str, Any]] = None):
        event = Event(name=event_name, learner=self, payload=payload or {})
        for cb in self.callbacks: cb(event)
        
        if self.current_task and hasattr(self.current_task, 'update_state'):
            self.current_task.update_state(state='PROGRESS', meta=payload)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, epochs: int):
        self.history = {"loss": []}
        X_train_t, y_train_t = Tensor(X_train), Tensor(y_train)
        
        # pipeline_name'i ilk başta alalım
        pipeline_name = "Bilinmiyor"
        if self.current_task and hasattr(self.current_task, 'request'):
            pipeline_name = self.current_task.request.kwargs.get('config', {}).get('pipeline_name', 'Bilinmiyor')

        self._publish("train_begin", payload={"total_epochs": epochs, "status_text": "Eğitim başlıyor...", "pipeline_name": pipeline_name})
        
        for epoch in range(epochs):
            if self.stop_training: break
            
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
            
            # KRİTİK DEĞİŞİKLİK: CPU'ya nefes alması için çok kısa bir süre ver.
            # Bu, worker'ın durum güncelleme mesajını göndermesine olanak tanır.
            time.sleep(0.01)
            
        self._publish("train_end", payload={"status_text": "Eğitim tamamlandı.", "pipeline_name": pipeline_name})
        return self.history