# ========== GÜNCELLENECEK DOSYA: src/azuraforge_learner/learner.py ==========
import pickle
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
        self.history: Dict[str, List[float]] = {}
        self.stop_training: bool = False

    def _publish(self, event_name: str, payload: Optional[Dict[str, Any]] = None):
        event = Event(name=event_name, learner=self, payload=payload or {})
        for cb in self.callbacks:
            cb(event)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, epochs: int, validation_data=None):
        self.history = {} # Her fit çağrısında geçmişi sıfırla
        X_train_t, y_train_t = Tensor(X_train), Tensor(y_train)
        
        self._publish("train_begin")
        for epoch in range(epochs):
            if self.stop_training:
                print(f"Stopping training at epoch {epoch} due to a callback.")
                break
            
            epoch_logs = {"epoch": epoch}
            self._publish("epoch_begin", payload=epoch_logs)
            
            # Eğitim adımı
            y_pred = self.model(X_train_t)
            loss = self.criterion(y_pred, y_train_t)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            epoch_logs["loss"] = loss.data.item()

            # Doğrulama adımı
            if validation_data:
                X_val, y_val = validation_data
                val_metrics = self.evaluate(X_val, y_val)
                epoch_logs.update(val_metrics)
            
            # Tarihçeyi kaydet
            for key, value in epoch_logs.items():
                if key not in self.history: self.history[key] = []
                self.history[key].append(value)
            
            self._publish("epoch_end", payload=epoch_logs)

        self._publish("train_end")
        return self.history

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return self.model(Tensor(X_test)).to_cpu()

    def evaluate(self, X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, float]:
        from sklearn.metrics import r2_score
        y_val_t = Tensor(y_val)
        y_pred = self.model(Tensor(X_val))
        val_loss = self.criterion(y_pred, y_val_t).data.item()
        
        # Şimdilik sadece regresyon metrikleri ekleyelim
        y_pred_np = y_pred.to_cpu()
        val_r2 = r2_score(y_val, y_pred_np)

        return {"val_loss": val_loss, "val_r2": val_r2}

    def save(self, filepath: str):
        # Basit bir kaydetme mekanizması (modelin parametrelerini kaydeder)
        with open(filepath, 'wb') as f:
            pickle.dump(self.model.parameters(), f)