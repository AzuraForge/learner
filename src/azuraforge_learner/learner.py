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
        for cb in self.callbacks:
            if hasattr(cb, 'set_learner'):
                 cb.set_learner(self)
                 
        self.history: Dict[str, List[float]] = {}
        self.stop_training: bool = False

    def _publish(self, event_name: str, payload: Optional[Dict[str, Any]] = None):
        event = Event(name=event_name, learner=self, payload=payload or {})
        for cb in self.callbacks:
            cb(event)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, epochs: int, pipeline_name: str = "Bilinmiyor"):
        self.history = {"loss": []}
        X_train_t, y_train_t = Tensor(X_train), Tensor(y_train)
        
        self._publish("train_begin", payload={"total_epochs": epochs, "status_text": "Eğitim başlıyor...", "pipeline_name": pipeline_name})
        
        for epoch in range(epochs):
            if self.stop_training:
                break
            
            self._publish("epoch_begin", payload={"epoch": epoch, "total_epochs": epochs, "pipeline_name": pipeline_name})
            
            # Eğitim adımı
            y_pred = self.model(X_train_t)
            loss = self.criterion(y_pred, y_train_t)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            current_loss = loss.to_cpu().item() if hasattr(loss, 'to_cpu') else float(loss.data)
            
            epoch_logs = {
                "epoch": epoch + 1, "total_epochs": epochs, "loss": current_loss,
                "status_text": f"Epoch {epoch + 1}/{epochs} tamamlandı, Kayıp: {current_loss:.6f}",
                "pipeline_name": pipeline_name
            }
            
            self.history["loss"].append(current_loss)
            self._publish("epoch_end", payload=epoch_logs)
            
        self._publish("train_end", payload={"status_text": "Eğitim tamamlandı.", "pipeline_name": pipeline_name})
        return self.history

    # YENİ: predict metodu eklendi
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Verilen girdi verisi (X_test) için model tahminleri yapar.
        """
        if not isinstance(X_test, np.ndarray):
            raise TypeError("Girdi (X_test) bir NumPy dizisi olmalıdır.")
        
        # Modeli Tensor'a çevir ve tahmin yap
        input_tensor = Tensor(X_test)
        predictions_tensor = self.model(input_tensor)
        
        # Sonucu CPU'ya çekip NumPy dizisi olarak döndür
        return predictions_tensor.to_cpu()

    # YENİ: evaluate metodu eklendi
    def evaluate(self, X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, float]:
        """
        Verilen doğrulama seti üzerinde modelin performansını değerlendirir.
        Şimdilik sadece regresyon metriklerini destekliyor.
        """
        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
        
        y_val_t = Tensor(y_val)
        y_pred_t = self.model(Tensor(X_val))
        
        val_loss = self.criterion(y_pred_t, y_val_t).to_cpu().item()
        y_pred_np = y_pred_t.to_cpu()
        
        # y_val'in NumPy olduğundan emin ol
        y_val_np = y_val if isinstance(y_val, np.ndarray) else np.array(y_val)
        
        val_r2 = r2_score(y_val_np, y_pred_np)
        val_mae = mean_absolute_error(y_val_np, y_pred_np)
        val_rmse = np.sqrt(mean_squared_error(y_val_np, y_pred_np))

        return {"val_loss": val_loss, "val_r2": val_r2, "val_mae": val_mae, "val_rmse": val_rmse}