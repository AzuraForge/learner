# ========== DOSYA: src/azuraforge_learner/learner.py ==========
from typing import Any, Dict, List, Optional
from .events import Event
from .models import Sequential
from .losses import Loss
from .optimizers import Optimizer
from .callbacks import Callback
from azuraforge_core import Tensor

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

    def fit(self, X_train_np, y_train_np, epochs: int):
        X_train, y_train = Tensor(X_train_np), Tensor(y_train_np)
        self._publish("train_begin")
        for epoch in range(epochs):
            if self.stop_training: break
            self._publish("epoch_begin", payload={"epoch": epoch})
            
            y_pred = self.model.forward(X_train)
            loss = self.criterion(y_pred, y_train)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            epoch_logs = {"loss": loss.data.item()}
            self._publish("epoch_end", payload=epoch_logs)
        self._publish("train_end")