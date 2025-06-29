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
    def __init__(self, model: Sequential, criterion: Loss, optimizer: Optimizer, callbacks: Optional[List[Callback]] = None, current_task: Optional[Any] = None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.callbacks = callbacks or []
        self.history: Dict[str, List[float]] = {}
        self.stop_training: bool = False
        self.current_task = current_task # Celery Task referansı

    def _publish(self, event_name: str, payload: Optional[Dict[str, Any]] = None):
        event = Event(name=event_name, learner=self, payload=payload or {})
        for cb in self.callbacks: cb(event)
        
        # --- KRİTİK DÜZELTME: Celery Task durumunu güncelle ---
        if self.current_task and hasattr(self.current_task, 'update_state'):
            # Celery'ye PROGRESS durumu ve meta verileri gönder
            # payload, epoch_logs'u içeriyor
            self.current_task.update_state(state='PROGRESS', meta=payload)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, epochs: int):
        self.history = {} # Her fit çağrısında geçmişi sıfırla
        
        # History'yi başlat
        for key in ["loss", "val_loss", "val_r2"]: # Ekleyeceğimiz metrikler için yer açalım
             self.history[key] = []

        X_train_t, y_train_t = Tensor(X_train), Tensor(y_train)
        
        self._publish("train_begin", payload={"total_epochs": epochs}) # Toplam epoch sayısını gönder
        for epoch in range(epochs):
            if self.stop_training: break
            
            # Epoch başlangıcı olayını yayınla
            self._publish("epoch_begin", payload={"epoch": epoch, "total_epochs": epochs})
            
            # Eğitim adımı
            y_pred = self.model(X_train_t)
            loss = self.criterion(y_pred, y_train_t)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            epoch_logs = {
                "epoch": epoch + 1, # Epoch sayısını 1'den başlatalım
                "total_epochs": epochs,
                "loss": loss.data.item(),
                "status_text": f"Epoch {epoch+1}/{epochs} completed..."
            }
            
            # History'ye ekle
            self.history["loss"].append(epoch_logs["loss"])
            
            # Epoch sonu olayını yayınla (payload olarak tüm epoch loglarını gönder)
            self._publish("epoch_end", payload=epoch_logs)
        self._publish("train_end")
        return self.history

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return self.model(Tensor(X_test)).to_cpu()

    def evaluate(self, X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, float]:
        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
        y_val_t = Tensor(y_val)
        y_pred = self.model(Tensor(X_val))
        
        val_loss = self.criterion(y_pred, y_val_t).data.item()
        y_pred_np = y_pred.to_cpu()
        
        val_r2 = r2_score(y_val, y_pred_np)
        val_mae = mean_absolute_error(y_val, y_pred_np)
        val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_np))

        return {"val_loss": val_loss, "val_r2": val_r2, "val_mae": val_mae, "val_rmse": val_rmse}
