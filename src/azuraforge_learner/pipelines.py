# learner/src/azuraforge_learner/pipelines.py

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from .learner import Learner, Callback
from .models import Sequential
from .reporting import generate_regression_report
from .optimizers import Adam, SGD
from .losses import MSELoss

def _create_sequences(data: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """Yardımcı fonksiyon: Veriyi (samples, sequence_length, features) formatına dönüştürür."""
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

class BasePipeline(ABC):
    """Tüm pipeline'lar için en temel soyut sınıf."""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        logging.basicConfig(level="INFO", format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    @abstractmethod
    def run(self, callbacks: Optional[List[Callback]] = None) -> Dict[str, Any]:
        pass

class TimeSeriesPipeline(BasePipeline):
    """
    Zaman serisi problemleri için standartlaştırılmış bir pipeline iskeleti.
    Veri işleme, eğitim, değerlendirme ve raporlama akışını yönetir.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.scaler = MinMaxScaler(feature_range=(-1, 1))

    @abstractmethod
    def _load_data(self) -> pd.DataFrame:
        """
        Ham veriyi (genellikle bir DataFrame) yüklemek için implemente edilmelidir.
        DataFrame'in indeksi bir zaman damgası olmalıdır.
        """
        pass

    @abstractmethod
    def _get_target_and_feature_cols(self) -> Tuple[str, List[str]]:
        """
        (hedef_sütun, özellik_sütunları_listesi) döndürmelidir.
        """
        pass
    
    @abstractmethod
    def _create_model(self, input_shape: Tuple) -> Sequential:
        """Uygulamaya özel modeli oluşturur."""
        pass

    def _create_learner(self, model: Sequential, callbacks: Optional[List[Callback]]) -> Learner:
        """
        Yapılandırmaya göre Learner'ı oluşturur. Genellikle override etmeye gerek yoktur.
        """
        training_params = self.config.get("training_params", {})
        lr = float(training_params.get("lr", 0.001))
        optimizer_type = str(training_params.get("optimizer", "adam")).lower()

        if optimizer_type == "adam":
            optimizer = Adam(model.parameters(), lr=lr)
        else:
            optimizer = SGD(model.parameters(), lr=lr)
            
        return Learner(model, MSELoss(), optimizer, callbacks=callbacks)

    def run(self, callbacks: Optional[List[Callback]] = None) -> Dict[str, Any]:
        """Deneyin tam yaşam döngüsünü çalıştırır."""
        self.logger.info(f"'{self.config.get('pipeline_name')}' pipeline başlatılıyor...")
        
        # 1. Veri Yükleme ve Hazırlama
        raw_data = self._load_data()
        target_col, feature_cols = self._get_target_and_feature_cols()
        
        if not all(col in raw_data.columns for col in feature_cols):
            raise ValueError("Yapılandırılan özellik sütunları veride bulunamadı.")
            
        data_to_process = raw_data[feature_cols]

        # 2. Veri Ölçekleme ve Sekanslara Ayırma
        sequence_length = self.config.get("model_params", {}).get("sequence_length", 60)
        scaled_data = self.scaler.fit_transform(data_to_process)
        
        if len(scaled_data) <= sequence_length:
            msg = "Sekans oluşturmak için yeterli veri yok."
            self.logger.warning(msg)
            return {"status": "failed", "message": msg}
        
        X, y_unsequenced = _create_sequences(scaled_data, sequence_length)
        
        # Hedef sütunun ölçeklenmiş verideki indeksini bul
        target_idx = feature_cols.index(target_col)
        y = y_unsequenced[:, target_idx].reshape(-1, 1) # Sadece hedef sütunu al
        
        # 3. Eğitim ve Test Setlerine Ayırma
        test_size = self.config.get("training_params", {}).get("test_size", 0.2)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        time_index_test = raw_data.index[split_idx + sequence_length:]

        # 4. Model ve Learner Oluşturma
        model = self._create_model(X_train.shape)
        learner = self._create_learner(model, callbacks)

        # 5. Eğitim
        epochs = int(self.config.get("training_params", {}).get("epochs", 50))
        self.logger.info(f"{epochs} epoch için model eğitimi başlıyor...")
        history = learner.fit(X_train, y_train, epochs=epochs, pipeline_name=self.config.get("pipeline_name"))
        
        # 6. Değerlendirme ve Sonuçları İşleme
        self.logger.info("Model değerlendiriliyor...")
        y_pred_scaled = learner.predict(X_test)
        
        # Tahminlerin ve gerçek değerlerin ölçeğini geri al
        y_pred_unscaled = self.scaler.inverse_transform(
            np.tile(y_pred_scaled, (1, len(feature_cols)))
        )[:, target_idx]
        y_test_unscaled = self.scaler.inverse_transform(
             np.tile(y_test, (1, len(feature_cols)))
        )[:, target_idx]

        from sklearn.metrics import r2_score, mean_absolute_error
        metrics = {
            'r2_score': r2_score(y_test_unscaled, y_pred_unscaled),
            'mae': mean_absolute_error(y_test_unscaled, y_pred_unscaled)
        }
        
        # 7. Raporlama
        self.logger.info("Rapor oluşturuluyor...")
        report_data = {
            "history": history, "metrics": metrics,
            "y_true": y_test_unscaled, "y_pred": y_pred_unscaled,
            "time_index": time_index_test, "y_label": target_col
        }
        generate_regression_report(report_data, self.config)
        
        return {
            "final_loss": history['loss'][-1] if history.get('loss') else None,
            "metrics": metrics
        }