import pickle
import time
import json # YENİ
import os   # YENİ
from typing import Any, Dict, List, Optional

import numpy as np
import redis # YENİ

from azuraforge_core import Tensor
from .events import Event
from .models import Sequential
from .losses import Loss
from .optimizers import Optimizer
from .callbacks import Callback

class Learner:
    def __init__(self, model: Sequential, criterion: Loss, optimizer: Optimizer, callbacks: Optional[List[Callback]] = None, task_id: Optional[str] = None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.callbacks = callbacks or []
        self.history: Dict[str, List[float]] = {}
        self.stop_training: bool = False
        
        # DÜZELTME: Artık Celery task objesini değil, sadece ID'sini ve Redis istemcisini tutuyoruz.
        self.task_id = task_id
        self._redis_client = None
        if self.task_id:
            try:
                redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
                self._redis_client = redis.from_url(redis_url)
            except Exception as e:
                print(f"HATA: Learner içinde Redis'e bağlanılamadı: {e}")

    def _publish(self, event_name: str, payload: Optional[Dict[str, Any]] = None):
        event = Event(name=event_name, learner=self, payload=payload or {})
        for cb in self.callbacks: cb(event)
        
        # DÜZELTME: Artık Celery state'i güncellemek yerine Redis Pub/Sub'a yayın yapıyoruz.
        if self._redis_client and self.task_id and payload:
            try:
                channel = f"task-progress:{self.task_id}"
                message = json.dumps(payload)
                self._redis_client.publish(channel, message)
            except Exception as e:
                # Eğitimi durdurmamak için hatayı sadece logluyoruz.
                print(f"HATA: Redis'e ilerleme durumu yayınlanamadı: {e}")

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, epochs: int, pipeline_name: str = "Bilinmiyor"):
        self.history = {"loss": []}
        X_train_t, y_train_t = Tensor(X_train), Tensor(y_train)
        
        self._publish("train_begin", payload={"status_text": "Eğitim başlıyor...", "pipeline_name": pipeline_name, "total_epochs": epochs})
        
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
            
        self._publish("train_end", payload={"status_text": "Eğitim tamamlandı.", "pipeline_name": pipeline_name})
        return self.history