# ========== DOSYA: learner/src/azuraforge_learner/learner.py ==========
import time
from typing import Any, Dict, List, Optional
import numpy as np

from azuraforge_core import Tensor, xp  # xp'yi import ediyoruz
from .events import Event
from .models import Sequential
from .losses import Loss
from .optimizers import Optimizer
from .callbacks import Callback

class Learner:
    def __init__(self, model: Sequential, criterion: Optional[Loss] = None, optimizer: Optional[Optimizer] = None, callbacks: Optional[List[Callback]] = None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.callbacks = callbacks or []
        
        for cb in self.callbacks:
            cb.set_learner(self)
                 
        self.history: Dict[str, List[float]] = {}
        self.stop_training: bool = False

    def _publish(self, event_name: str, payload: Optional[Dict[str, Any]] = None):
        event = Event(name=event_name, learner=self, payload=payload or {})
        for cb in self.callbacks:
            cb(event)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, epochs: int, batch_size: int, pipeline_name: str = "Bilinmiyor"):
        if not self.criterion or not self.optimizer:
            raise RuntimeError("Cannot fit the model without a criterion and an optimizer.")
            
        self.history = {"loss": []}
        
        self._publish("train_begin", payload={"total_epochs": epochs, "status_text": "Eğitim başlıyor...", "pipeline_name": pipeline_name})
        
        num_samples = X_train.shape[0]

        for epoch in range(epochs):
            if self.stop_training:
                break
            
            # === YENİ: Veriyi karıştırarak batch'lere hazırlama ===
            permutation = np.random.permutation(num_samples)
            X_shuffled = X_train[permutation]
            y_shuffled = y_train[permutation]
            
            epoch_losses = []
            self._publish("epoch_begin", payload={"epoch": epoch, "total_epochs": epochs, "pipeline_name": pipeline_name})
            
            # === YENİ: Mini-batch döngüsü ===
            for i in range(0, num_samples, batch_size):
                self._publish("batch_begin", payload={"epoch": epoch, "batch_index": i // batch_size})

                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]

                X_batch_t, y_batch_t = Tensor(X_batch), Tensor(y_batch)

                # İleri geçiş
                y_pred = self.model(X_batch_t)
                loss = self.criterion(y_pred, y_batch_t)
                
                # Geriye yayılım ve optimizasyon
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Bu batch'in kaybını kaydet
                epoch_losses.append(loss.to_cpu().item())
                self._publish("batch_end", payload={"epoch": epoch, "batch_index": i // batch_size, "batch_loss": loss.to_cpu().item()})

            # Epoch sonu işlemleri
            current_loss = np.mean(epoch_losses)
            self.history["loss"].append(current_loss)
            
            epoch_logs = {
                "epoch": epoch + 1, "total_epochs": epochs, "loss": current_loss,
                "status_text": f"Epoch {epoch + 1}/{epochs} tamamlandı, Kayıp: {current_loss:.6f}",
                "pipeline_name": pipeline_name
            }
            
            self._publish("epoch_end", payload=epoch_logs)
            
        self._publish("train_end", payload={"status_text": "Eğitim tamamlandı.", "pipeline_name": pipeline_name})
        return self.history
        
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        if not isinstance(X_test, np.ndarray):
            raise TypeError("Girdi (X_test) bir NumPy dizisi olmalıdır.")
        
        # Modeli tahmin moduna al (örn: Dropout'u kapatmak için)
        self.model.eval()
        
        batch_size = 128
        num_samples = X_test.shape[0]
        all_predictions = []

        for i in range(0, num_samples, batch_size):
            X_batch = X_test[i:i+batch_size]
            input_tensor = Tensor(X_batch)
            predictions_tensor = self.model(input_tensor)
            all_predictions.append(predictions_tensor.to_cpu())
            
        # Tahmin bittikten sonra modeli tekrar eğitim moduna alabiliriz, bu iyi bir pratik.
        self.model.train()

        return np.vstack(all_predictions) if all_predictions else np.array([])

    def evaluate(self, X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, float]:
        if not self.criterion:
            raise RuntimeError("Cannot evaluate the model without a criterion.")
            
        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
        
        # predict metodu zaten batch'ler halinde çalışıyor
        y_pred_np = self.predict(X_val)

        # y_val'in de (N, 1) şeklinde olduğundan emin olalım
        y_val_np = y_val.reshape(-1, 1) if isinstance(y_val, np.ndarray) else np.array(y_val).reshape(-1, 1)

        # Kaybı hesaplamak için tensörlere çevir
        y_val_t = Tensor(y_val_np)
        y_pred_t = Tensor(y_pred_np)
        val_loss = self.criterion(y_pred_t, y_val_t).to_cpu().item()
        
        val_r2 = r2_score(y_val_np, y_pred_np)
        val_mae = mean_absolute_error(y_val_np, y_pred_np)
        val_rmse = np.sqrt(mean_squared_error(y_val_np, y_pred_np))

        return {"val_loss": val_loss, "val_r2": val_r2, "val_mae": val_mae, "val_rmse": val_rmse}

    def save_model(self, filepath: str):
        self.model.save(filepath)

    def load_model(self, filepath: str):
        self.model.load(filepath)