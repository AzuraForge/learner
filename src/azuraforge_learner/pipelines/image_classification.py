# learner/src/azuraforge_learner/pipelines/image_classification.py
"""
Görüntü sınıflandırma pipeline'ları için temel sınıfı içerir.
"""
# DÜZELTME: Gerekli tüm importlar eklendi.
from abc import abstractmethod
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
from pydantic import BaseModel
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from .base import BasePipeline
from ..callbacks import Callback
from ..learner import Learner
from ..losses import CrossEntropyLoss
from ..models import Sequential
from ..optimizers import Adam
from ..reporting import generate_classification_report

class ImageClassificationPipeline(BasePipeline):
    """Görüntü sınıflandırma görevleri için temel pipeline."""
    
    def get_config_model(self) -> Optional[type[BaseModel]]:
        return None

    @abstractmethod
    def _load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        pass

    @abstractmethod
    def _create_model(self, input_shape: Tuple, num_classes: int) -> Sequential:
        pass

    def _create_learner(self, model: Sequential, callbacks: Optional[List[Callback]]) -> Learner:
        training_params = self.config.get("training_params", {})
        lr = float(training_params.get("lr", 0.001))
        optimizer = Adam(model.parameters(), lr=lr)
        return Learner(model, CrossEntropyLoss(), optimizer, callbacks=callbacks)

    def run(self, callbacks: Optional[List[Callback]] = None, skip_training: bool = False) -> Dict[str, Any]:
        self.logger.info(f"Running ImageClassificationPipeline: '{self.config.get('pipeline_name')}'...")
        
        X_train, y_train, X_test, y_test, class_names = self._load_data()
        num_classes = len(class_names)
        
        model = self._create_model(X_train.shape, num_classes)
        self.learner = self._create_learner(model, callbacks or [])
        
        epochs = int(self.config.get("training_params", {}).get("epochs", 10))
        history = self.learner.fit(X_train, y_train, epochs=epochs, pipeline_name=self.config.get("pipeline_name"))
        
        self.logger.info("Evaluating model on test set...")
        y_pred_logits = self.learner.predict(X_test)
        y_pred_labels = np.argmax(y_pred_logits, axis=1)
        
        accuracy = accuracy_score(y_test, y_pred_labels)
        self.logger.info(f"Test Accuracy: {accuracy:.4f}")
        
        report = classification_report(y_test, y_pred_labels, target_names=class_names, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred_labels)
        
        final_results = {
            "history": history,
            "metrics": { "accuracy": accuracy, "classification_report": report },
            "confusion_matrix": conf_matrix.tolist()
        }
        
        generate_classification_report(final_results, self.config, class_names)
        return final_results