# learner/src/azuraforge_learner/pipelines.py

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .learner import Learner, Callback
from .models import Sequential
from .utils import generate_regression_report

class BasePipeline(ABC):
    """
    Tüm AzuraForge pipeline eklentileri için temel soyut sınıf.
    Bir deneyin standart yaşam döngüsünü yönetir.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.scaler = StandardScaler()

    @abstractmethod
    def _load_data(self) -> Any:
        """Uygulamaya özel veri yükleme mantığı."""
        pass

    @abstractmethod
    def _preprocess(self, data: Any) -> Tuple[np.ndarray, np.ndarray]:
        """Veriyi X ve y olarak ayırıp ön işleme tabi tutar."""
        pass
    
    @abstractmethod
    def _create_model(self, input_shape: Tuple) -> Sequential:
        """Uygulamaya özel modeli oluşturur."""
        pass

    @abstractmethod
    def _create_learner(self, model: Sequential, callbacks: Optional[List[Callback]]) -> Learner:
        """Uygulamaya özel Learner ve Optimizer'ı oluşturur."""
        pass
    
    def _postprocess_and_evaluate(self, learner: Learner, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Tahminleri yapar, ölçeklemeyi geri alır ve metrikleri hesaplar.
        Varsayılan implementasyon regresyon içindir. Gerekirse override edilebilir.
        """
        from sklearn.metrics import r2_score, mean_absolute_error

        y_pred = learner.predict(X_test)
        
        # Varsayılan olarak y'nin tek bir sütun olduğunu varsayıyoruz
        # Ölçeklemeyi geri almak için, scaler'ın n_features_in_ özelliğini kullanıyoruz
        if hasattr(self.scaler, 'n_features_in_'):
            dummy_test = np.zeros((len(y_test), self.scaler.n_features_in_))
            dummy_pred = np.zeros((len(y_pred), self.scaler.n_features_in_))

            dummy_test[:, -1] = y_test.flatten()
            dummy_pred[:, -1] = y_pred.flatten()

            y_test_unscaled = self.scaler.inverse_transform(dummy_test)[:, -1]
            y_pred_unscaled = self.scaler.inverse_transform(dummy_pred)[:, -1]
        else:
            # Scaler hiç kullanılmamışsa
            y_test_unscaled = y_test
            y_pred_unscaled = y_pred

        metrics = {
            'r2_score': r2_score(y_test_unscaled, y_pred_unscaled),
            'mae': mean_absolute_error(y_test_unscaled, y_pred_unscaled)
        }
        
        return {
            "metrics": metrics,
            "y_true": y_test_unscaled,
            "y_pred": y_pred_unscaled
        }

    def run(self, callbacks: Optional[List[Callback]] = None) -> Dict[str, Any]:
        """Deneyin tam yaşam döngüsünü çalıştırır."""
        self.logger.info("Pipeline başlatılıyor...")
        
        raw_data = self._load_data()
        X, y = self._preprocess(raw_data)

        # Veriyi ölçekle ve böl
        X_scaled = self.scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, shuffle=False
        )

        model = self._create_model(X_train.shape)
        learner = self._create_learner(model, callbacks)

        self.logger.info("Model eğitimi başlıyor...")
        history = learner.fit(X_train, y_train.reshape(-1, 1))
        
        self.logger.info("Model değerlendiriliyor...")
        final_results = self._postprocess_and_evaluate(learner, X_test, y_test)
        final_results['history'] = history

        self.logger.info("Rapor oluşturuluyor...")
        generate_regression_report(final_results, self.config)
        
        # Worker'a döndürülecek nihai sonuç
        return {
            "final_loss": history['loss'][-1],
            "metrics": final_results['metrics']
        }