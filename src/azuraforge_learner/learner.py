# ========== GÜNCELLENMİŞ DOSYA: src/azuraforge_learner/learner.py ==========
import pickle
from typing import Any, Dict, List, Optional
from .events import Event
from .models import Sequential
from .losses import Loss
from .optimizers import Optimizer
from .callbacks import Callback

class Learner:
    """
    Modeli, veri kümesini, kayıp fonksiyonunu ve optimizatörü bir araya getiren
    ve eğitim döngüsünü yöneten ana sınıf. Olay tabanlı bir sistem kullanır.
    """
    def __init__(
        self,
        model: Sequential,
        criterion: Loss,
        optimizer: Optimizer,
        callbacks: Optional[List[Callback]] = None
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.callbacks = callbacks if callbacks else []
        self.history: Dict[str, List[float]] = {}
        self.stop_training: bool = False

        # Her callback'e learner referansını ver
        for cb in self.callbacks:
            cb.set_learner(self) # Bu satır aslında gereksiz kalıyor, event nesnesi learner'ı taşıyacak

    def _publish(self, event_name: str, payload: Optional[Dict[str, Any]] = None):
        """Abone olan tüm callback'lere bir olay yayınlar."""
        event = Event(name=event_name, learner=self, payload=payload or {})
        for cb in self.callbacks:
            cb(event)

    def fit(self, X_train, y_train, epochs: int, validation_data=None):
        self.history = {"loss": [], "val_loss": [], "val_accuracy": []}
        
        self._publish("train_begin")

        for epoch in range(epochs):
            if self.stop_training:
                break
            
            epoch_logs = {}
            self._publish("epoch_begin", payload={"epoch": epoch})

            # --- Eğitim Adımı ---
            # (Buraya batch'ler halinde eğitim döngüsü gelecek)
            y_pred = self.model.forward(X_train)
            loss = self.criterion(y_pred, y_train)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            epoch_logs["loss"] = loss.item() # .item() ile değeri al

            # --- Doğrulama Adımı ---
            if validation_data:
                X_val, y_val = validation_data
                val_metrics = self.evaluate(X_val, y_val, verbose=False)
                epoch_logs.update(val_metrics)
            
            # Tarihçeyi kaydet
            for key, value in epoch_logs.items():
                if key not in self.history: self.history[key] = []
                self.history[key].append(value)
            
            self._publish("epoch_end", payload=epoch_logs)

        self._publish("train_end")
        return self.history

    def evaluate(self, X_test, y_test, verbose=True) -> Dict[str, float]:
        # ... (Değerlendirme mantığı)
        return {"val_loss": ..., "val_accuracy": ...}

    def save(self, filepath: str):
        # ... (Model kaydetme mantığı)
        pass