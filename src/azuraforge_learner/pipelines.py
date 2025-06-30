# learner/src/azuraforge_learner/pipelines.py

import logging
import os
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
from .events import Event
from .caching import get_cache_filepath, load_from_cache, save_to_cache

def _create_sequences(data: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

class BasePipeline(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def run(self, callbacks: Optional[List[Callback]] = None) -> Dict[str, Any]:
        pass

class LivePredictionCallback(Callback):
    def __init__(self, pipeline: 'TimeSeriesPipeline', X_val: np.ndarray, y_val: np.ndarray, time_index_val: pd.Index, validate_every_n_epochs: int):
        super().__init__()
        self.pipeline = pipeline
        self.X_val = X_val
        self.y_val = y_val
        self.time_index_val = time_index_val
        self.validate_every = validate_every_n_epochs
        self.last_results: Dict[str, Any] = {}

    def on_epoch_end(self, event: Event) -> None:
        epoch = event.payload.get("epoch", 0)
        total_epochs = event.payload.get("total_epochs", 1)

        if (epoch % self.validate_every == 0 and epoch > 0) or (epoch == total_epochs):
            if not self.learner: return

            y_pred_scaled = self.learner.predict(self.X_val)
            
            target_col, feature_cols = self.pipeline._get_target_and_feature_cols()
            target_idx = feature_cols.index(target_col)
            
            dummy_pred = np.zeros((len(y_pred_scaled), len(feature_cols)))
            dummy_test = np.zeros((len(self.y_val), len(feature_cols)))
            dummy_pred[:, target_idx] = y_pred_scaled.flatten()
            dummy_test[:, target_idx] = self.y_val.flatten()
            y_pred_unscaled_transformed = self.pipeline.scaler.inverse_transform(dummy_pred)[:, target_idx]
            y_test_unscaled_transformed = self.pipeline.scaler.inverse_transform(dummy_test)[:, target_idx]

            # YENİ: Değerlendirme sırasında da ters dönüşümü uygula
            target_transform = self.pipeline.config.get("feature_engineering", {}).get("target_col_transform")
            if target_transform == 'log':
                y_pred_unscaled = np.exp(y_pred_unscaled_transformed)
                y_test_unscaled = np.exp(y_test_unscaled_transformed)
            else:
                y_pred_unscaled = y_pred_unscaled_transformed
                y_test_unscaled = y_test_unscaled_transformed

            validation_payload = {
                "x_axis": [d.isoformat() for d in self.time_index_val],
                "y_true": y_test_unscaled.tolist(), "y_pred": y_pred_unscaled.tolist(),
                "x_label": "Tarih", "y_label": target_col
            }
            event.payload['validation_data'] = validation_payload
            
            from sklearn.metrics import r2_score, mean_absolute_error
            self.last_results = {
                "history": self.learner.history,
                "metrics": {
                    'r2_score': r2_score(y_test_unscaled, y_pred_unscaled),
                    'mae': mean_absolute_error(y_test_unscaled, y_pred_unscaled)
                },
                "y_true": y_test_unscaled, "y_pred": y_pred_unscaled,
                "time_index": self.time_index_val, "y_label": target_col
            }


class TimeSeriesPipeline(BasePipeline):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.learner: Optional[Learner] = None

    @abstractmethod
    def _load_data_from_source(self) -> pd.DataFrame:
        pass
        
    def get_caching_params(self) -> Dict[str, Any]:
        return self.config.get("data_sourcing", {})

    @abstractmethod
    def _get_target_and_feature_cols(self) -> Tuple[str, List[str]]:
        pass
    
    @abstractmethod
    def _create_model(self, input_shape: Tuple) -> Sequential:
        pass

    def _create_learner(self, model: Sequential, callbacks: Optional[List[Callback]]) -> Learner:
        training_params = self.config.get("training_params", {})
        lr = float(training_params.get("lr", 0.001))
        optimizer_type = str(training_params.get("optimizer", "adam")).lower()
        optimizer = Adam(model.parameters(), lr=lr) if optimizer_type == "adam" else SGD(model.parameters(), lr=lr)
        return Learner(model, MSELoss(), optimizer, callbacks=callbacks)

    def run(self, callbacks: Optional[List[Callback]] = None) -> Dict[str, Any]:
        self.logger.info(f"'{self.config.get('pipeline_name')}' pipeline başlatılıyor...")
        
        system_config = self.config.get("system", {})
        cache_enabled = system_config.get("caching_enabled", True)
        cache_dir = os.getenv("CACHE_DIR", system_config.get("cache_dir", ".cache"))
        cache_max_age = system_config.get("cache_max_age_hours", 24)
        
        cache_params = self.get_caching_params()
        cache_filepath = get_cache_filepath(cache_dir, self.config.get('pipeline_name', 'default_context'), cache_params)

        raw_data = None
        if cache_enabled:
            raw_data = load_from_cache(cache_filepath, cache_max_age)

        if raw_data is None:
            self.logger.info("Önbellek boş veya geçersiz. Veri kaynaktan çekiliyor...")
            raw_data = self._load_data_from_source()
            if cache_enabled and isinstance(raw_data, pd.DataFrame) and not raw_data.empty:
                save_to_cache(raw_data, cache_filepath)
        
        target_col, feature_cols = self._get_target_and_feature_cols()
        
        if not all(col in raw_data.columns for col in feature_cols):
            raise ValueError(f"Yapılandırılan özellik sütunları ({feature_cols}) indirilen veride bulunamadı.")
            
        data_to_process = raw_data[feature_cols].copy() # .copy() ile warning'i engelle

        # YENİ: Logaritmik dönüşümü burada uygula
        fe_config = self.config.get("feature_engineering", {})
        target_transform = fe_config.get("target_col_transform")
        if target_transform == 'log':
            self.logger.info(f"'{target_col}' sütununa logaritmik dönüşüm uygulanıyor.")
            # Negatif veya sıfır değerlerden kaçınmak için 1 ekleyebilir veya klipleyebiliriz.
            data_to_process[target_col] = np.log1p(data_to_process[target_col])
        
        sequence_length = self.config.get("model_params", {}).get("sequence_length", 60)
        scaled_data = self.scaler.fit_transform(data_to_process)
        
        if len(scaled_data) <= sequence_length:
            return {"status": "failed", "message": "Sekans oluşturmak için yeterli veri yok."}
        
        X, y_unsequenced = _create_sequences(scaled_data, sequence_length)
        
        target_idx = feature_cols.index(target_col)
        y = y_unsequenced[:, target_idx].reshape(-1, 1)
        
        test_size = self.config.get("training_params", {}).get("test_size", 0.2)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        time_index_test = raw_data.index[split_idx + sequence_length:]

        model = self._create_model(X_train.shape)
        
        live_predict_cb = LivePredictionCallback(
            pipeline=self, X_val=X_test, y_val=y_test, 
            time_index_val=time_index_test,
            validate_every_n_epochs=self.config.get("training_params", {}).get("validate_every", 5)
        )
        all_callbacks = (callbacks or []) + [live_predict_cb]
        
        self.learner = self._create_learner(model, all_callbacks)

        epochs = int(self.config.get("training_params", {}).get("epochs", 50))
        self.logger.info(f"{epochs} epoch için model eğitimi başlıyor...")
        history = self.learner.fit(X_train, y_train, epochs=epochs, pipeline_name=self.config.get("pipeline_name"))
        
        final_results = live_predict_cb.last_results
        if not final_results:
            return {"status": "failed", "message": "Eğitim tamamlanamadı veya hiç doğrulama yapılmadı."}

        self.logger.info("Rapor oluşturuluyor...")
        generate_regression_report(final_results, self.config)
        
        final_loss = history['loss'][-1] if history.get('loss') else None
        
        # Sonucu worker'a döndürürken de ters dönüşüm yapıldığından emin olmalıyız.
        # LivePredictionCallback bunu zaten yapıyor. Ondan gelen sonucu kullanalım.
        return {
            "final_loss": final_loss,
            "metrics": final_results.get('metrics', {}),
            "history": final_results.get('history', {}),
            "y_true": final_results.get('y_true', np.array([])).tolist(),
            "y_pred": final_results.get('y_pred', np.array([])).tolist(),
            "time_index": [d.isoformat() for d in final_results.get('time_index', [])]
        }