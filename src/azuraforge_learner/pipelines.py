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
    return np.array(xs), np.array(ys).reshape(-1, data.shape[1])

class BasePipeline(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def run(self, callbacks: Optional[List[Callback]] = None) -> Dict[str, Any]:
        pass

class LivePredictionCallback(Callback):
    def __init__(self, pipeline: 'TimeSeriesPipeline', X_val: np.ndarray, y_val: np.ndarray, time_index_val: pd.Index):
        super().__init__()
        self.pipeline = pipeline
        self.X_val = X_val
        self.y_val = y_val
        self.time_index_val = time_index_val
        # Düzeltme: validate_every config'den gelmeli, yoksa varsayılan 5.
        self.validate_every = self.pipeline.config.get("training_params", {}).get("validate_every", 5)
        self.last_results: Dict[str, Any] = {}

    def on_epoch_end(self, event: Event) -> None:
        epoch = event.payload.get("epoch", 0)
        total_epochs = event.payload.get("total_epochs", 1)

        # BURADAKİ DÜZELTME: validate_every kontrolü.
        # Eğer validate_every > 0 ise, sadece o aralıklarda ve son epoch'ta gönder.
        # Eğer validate_every <= 0 ise, her epoch'ta gönder (devre dışı bırakma gibi).
        should_validate_and_send = False
        if self.validate_every > 0:
            if (epoch % self.validate_every == 0 and epoch > 0) or \
               (epoch == total_epochs and total_epochs > 0): # Son epoch ise her zaman gönder
                should_validate_and_send = True
        else: # validate_every 0 veya negatifse, her epoch'ta gönder
            should_validate_and_send = True

        if should_validate_and_send:
            if not self.learner: return

            y_pred_scaled = self.learner.predict(self.X_val)
            
            y_test_unscaled, y_pred_unscaled = self.pipeline._inverse_transform_all(
                self.y_val, y_pred_scaled
            )
            
            # validation_data'yı payload'a ekliyoruz.
            event.payload['validation_data'] = {
                "x_axis": [d.isoformat() for d in self.time_index_val],
                "y_true": y_test_unscaled.tolist(), 
                "y_pred": y_pred_unscaled.tolist(), 
                "x_label": "Tarih", 
                "y_label": self.pipeline._get_target_and_feature_cols()[0]
            }
            
            # Metrikleri hesapla ve sakla (API'ye nihai results olarak gitmek için)
            from sklearn.metrics import r2_score, mean_absolute_error
            self.last_results = {
                "history": self.learner.history,
                "metrics": {
                    'r2_score': float(r2_score(y_test_unscaled, y_pred_unscaled)), 
                    'mae': float(mean_absolute_error(y_test_unscaled, y_pred_unscaled))
                },
                "final_loss": event.payload.get("loss"), 
                "y_true": y_test_unscaled.tolist(),
                "y_pred": y_pred_unscaled.tolist(),
                "time_index": [d.isoformat() for d in self.time_index_val],
                "y_label": self.pipeline._get_target_and_feature_cols()[0]
            }

class TimeSeriesPipeline(BasePipeline):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.feature_scaler = MinMaxScaler(feature_range=(-1, 1))
        self.learner: Optional[Learner] = None
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.X_test: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None
        self.time_index_test: Optional[pd.Index] = None

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

    def _inverse_transform_all(self, y_true_scaled, y_pred_scaled):
        y_true_unscaled_transformed = self.scaler.inverse_transform(y_true_scaled)
        y_pred_unscaled_transformed = self.scaler.inverse_transform(y_pred_scaled)

        target_transform = self.config.get("feature_engineering", {}).get("target_col_transform")
        if target_transform == 'log':
            self.logger.info(f"Target column will be exponentiated from log-transformed data.")
            y_true_final = np.expm1(y_true_unscaled_transformed)
            y_pred_final = np.expm1(y_pred_unscaled_transformed)
        else:
            y_true_final = y_true_unscaled_transformed
            y_pred_final = y_pred_unscaled_transformed
            
        return y_true_final.flatten(), y_pred_final.flatten()

    def run(self, callbacks: Optional[List[Callback]] = None) -> Dict[str, Any]:
        self.logger.info(f"'{self.config.get('pipeline_name')}' pipeline başlatılıyor...")
        
        system_config = self.config.get("system", {})
        cache_enabled = system_config.get("caching_enabled", True)
        cache_dir = os.getenv("CACHE_DIR", ".cache")
        cache_max_age = system_config.get("cache_max_age_hours", 24)
        
        cache_params = self.get_caching_params()
        cache_filepath = get_cache_filepath(cache_dir, self.config.get('pipeline_name', 'default_context'), cache_params)

        raw_data = None
        if cache_enabled: raw_data = load_from_cache(cache_filepath, cache_max_age)
        if raw_data is None:
            self.logger.info("Önbellek boş veya geçersiz. Veri kaynaktan çekiliyor...")
            raw_data = self._load_data_from_source()
            if cache_enabled and isinstance(raw_data, pd.DataFrame) and not raw_data.empty:
                save_to_cache(raw_data, cache_filepath)

        target_col, feature_cols = self._get_target_and_feature_cols()
        
        features_df = raw_data[feature_cols].copy()
        target_series = raw_data[target_col].copy()

        target_transform = self.config.get("feature_engineering", {}).get("target_col_transform")
        if target_transform == 'log':
            self.logger.info(f"'{target_col}' sütununa log(1+x) dönüşümü uygulanıyor.")
            target_series = np.log1p(target_series)
        
        scaled_features = self.feature_scaler.fit_transform(features_df)
        scaled_target = self.scaler.fit_transform(target_series.values.reshape(-1, 1))
        
        scaled_data = np.concatenate([scaled_features, scaled_target], axis=1)

        sequence_length = self.config.get("model_params", {}).get("sequence_length", 60)
        if len(scaled_data) <= sequence_length:
            return {"status": "failed", "message": "Sekans oluşturmak için yeterli veri yok."}
        
        X, y_unsequenced = _create_sequences(scaled_data, sequence_length)
        
        target_idx = feature_cols.index(target_col)
        y = y_unsequenced[:, target_idx].reshape(-1, 1)
        
        test_size = self.config.get("training_params", {}).get("test_size", 0.2)
        split_idx = int(len(X) * (1 - test_size))
        self.X_train, self.X_test = X[:split_idx], X[split_idx:]
        self.y_train, self.y_test = y[:split_idx], y[split_idx:]
        self.time_index_test = raw_data.index[split_idx + sequence_length:]

        model = self._create_model(self.X_train.shape)
        
        live_predict_cb = LivePredictionCallback(pipeline=self, X_val=self.X_test, y_val=self.y_test, time_index_val=self.time_index_test)
        all_callbacks = (callbacks or []) + [live_predict_cb]
        
        self.learner = self._create_learner(model, all_callbacks)

        epochs = int(self.config.get("training_params", {}).get("epochs", 50))
        self.logger.info(f"{epochs} epoch için model eğitimi başlıyor...")
        history = self.learner.fit(self.X_train, self.y_train, epochs=epochs, pipeline_name=self.config.get("pipeline_name"))
        
        final_results = live_predict_cb.last_results
        if not final_results:
            self.logger.warning("Eğitim tamamlandı ancak LivePredictionCallback'den sonuç alınamadı. Manuel olarak toplanıyor.")
            final_results = {
                "history": history,
                "final_loss": history['loss'][-1] if history.get('loss') else None,
                "metrics": {},
                "y_true": [], "y_pred": [], "time_index": []
            }


        self.logger.info("Rapor oluşturuluyor...")
        # Raporlamayı sadece gerekli verilerle çağır
        generate_regression_report(
            {
                "metrics": final_results.get('metrics', {}),
                "history": final_results.get('history', {}),
                "y_true": final_results.get('y_true', []),
                "y_pred": final_results.get('y_pred', []),
                "time_index": final_results.get('time_index', []),
                "y_label": final_results.get('y_label', self._get_target_and_feature_cols()[0])
            },
            self.config
        )
        
        return {
            "final_loss": final_results.get('final_loss'),
            "metrics": final_results.get('metrics', {}),
            "history": final_results.get('history', {}), 
            "y_true": final_results.get('y_true', []),
            "y_pred": final_results.get('y_pred', []),
            "time_index": final_results.get('time_index', [])
        }