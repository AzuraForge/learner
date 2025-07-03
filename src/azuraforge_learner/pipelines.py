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
    return np.array(xs), np.array(ys).reshape(-1, data.shape[1] if data.ndim > 1 else -1)

class BasePipeline(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def run(self, callbacks: Optional[List[Callback]] = None, skip_training: bool = False) -> Dict[str, Any]:
        pass

class LivePredictionCallback(Callback):
    # ... Bu sınıfın içeriği aynı kalıyor, değişiklik yok ...
    def __init__(self, pipeline: 'TimeSeriesPipeline', X_val: np.ndarray, y_val: np.ndarray, time_index_val: pd.Index):
        super().__init__()
        self.pipeline = pipeline
        self.X_val = X_val
        self.y_val = y_val
        self.time_index_val = time_index_val
        self.validate_every = self.pipeline.config.get("training_params", {}).get("validate_every", 5)
        self.last_results: Dict[str, Any] = {}

    def on_epoch_end(self, event: Event) -> None:
        epoch = event.payload.get("epoch", 0)
        total_epochs = event.payload.get("total_epochs", 1)
        should_validate = (epoch % self.validate_every == 0) or (epoch == total_epochs) or (epoch == 1)
        if not should_validate: return
        if not self.learner: return
        try:
            y_pred_scaled = self.learner.predict(self.X_val)
            y_test_unscaled, y_pred_unscaled = self.pipeline._inverse_transform_all(self.y_val, y_pred_scaled)
            validation_data = {
                "x_axis": [d.isoformat() for d in self.time_index_val],
                "y_true": y_test_unscaled.tolist(), "y_pred": y_pred_unscaled.tolist(),
                "x_label": "Tarih", "y_label": self.pipeline.target_col
            }
            event.payload['validation_data'] = validation_data
            from sklearn.metrics import r2_score, mean_absolute_error
            self.last_results = {
                "history": self.learner.history,
                "metrics": {'r2_score': float(r2_score(y_test_unscaled, y_pred_unscaled)), 'mae': float(mean_absolute_error(y_test_unscaled, y_pred_unscaled))},
                "final_loss": event.payload.get("loss"), "y_true": y_test_unscaled.tolist(), "y_pred": y_pred_unscaled.tolist(),
                "time_index": [d.isoformat() for d in self.time_index_val], "y_label": self.pipeline.target_col
            }
        except Exception as e:
            logging.error(f"LivePredictionCallback Error: {e}", exc_info=True)


class TimeSeriesPipeline(BasePipeline):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.feature_scaler = MinMaxScaler(feature_range=(-1, 1))
        self.learner: Optional[Learner] = None
        self.target_col: Optional[str] = None
        self.feature_cols: Optional[List[str]] = None
        self.is_fitted: bool = False # Scaler'ların eğitilip eğitilmediğini takip eder

    @abstractmethod
    def _load_data_from_source(self) -> pd.DataFrame: pass
    def get_caching_params(self) -> Dict[str, Any]: return self.config.get("data_sourcing", {})
    @abstractmethod
    def _get_target_and_feature_cols(self) -> Tuple[str, List[str]]: pass
    @abstractmethod
    def _create_model(self, input_shape: Tuple) -> Sequential: pass

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
            y_true_final, y_pred_final = np.expm1(y_true_unscaled_transformed), np.expm1(y_pred_unscaled_transformed)
        else:
            y_true_final, y_pred_final = y_true_unscaled_transformed, y_pred_unscaled_transformed
        return y_true_final.flatten(), y_pred_final.flatten()

    def _prepare_scalers(self, raw_data: pd.DataFrame):
        """Scaler'ları eğitir ve durumu `is_fitted` olarak ayarlar."""
        if self.is_fitted: return
        self.logger.info("Preparing and fitting scalers...")
        self.target_col, self.feature_cols = self._get_target_and_feature_cols()
        features_df = raw_data[self.feature_cols].copy()
        target_series = raw_data[self.target_col].copy()
        if self.config.get("feature_engineering", {}).get("target_col_transform") == 'log':
            target_series = np.log1p(target_series)
        self.feature_scaler.fit(features_df)
        self.scaler.fit(target_series.values.reshape(-1, 1))
        self.is_fitted = True
        self.logger.info("Scalers have been fitted.")

    def run(self, callbacks: Optional[List[Callback]] = None, skip_training: bool = False) -> Dict[str, Any]:
        self.logger.info(f"'{self.config.get('pipeline_name')}' pipeline running...")
        
        system_config = self.config.get("system", {})
        cache_dir = os.getenv("CACHE_DIR", ".cache")
        raw_data = None
        if system_config.get("caching_enabled", True):
            cache_filepath = get_cache_filepath(cache_dir, self.config.get('pipeline_name', 'default'), self.get_caching_params())
            raw_data = load_from_cache(cache_filepath, system_config.get("cache_max_age_hours", 24))
        if raw_data is None:
            raw_data = self._load_data_from_source()
            if system_config.get("caching_enabled", True) and isinstance(raw_data, pd.DataFrame) and not raw_data.empty:
                save_to_cache(raw_data, cache_filepath)

        self._prepare_scalers(raw_data)
        if skip_training:
            self.logger.info("Skipping training. Pipeline run was for data preparation only.")
            return {"status": "skipped", "message": "Training was skipped."}
        
        target_series_processed = np.log1p(raw_data[self.target_col]) if self.config.get("feature_engineering", {}).get("target_col_transform") == 'log' else raw_data[self.target_col]
        scaled_features = self.feature_scaler.transform(raw_data[self.feature_cols])
        scaled_target = self.scaler.transform(target_series_processed.values.reshape(-1, 1))
        
        scaled_data = np.concatenate([scaled_features, scaled_target], axis=1)
        sequence_length = self.config.get("model_params", {}).get("sequence_length", 60)
        
        X, y_unsequenced = _create_sequences(scaled_data, sequence_length)
        if len(X) == 0: raise ValueError(f"Not enough data to create sequences. Need > {sequence_length} rows.")
        
        target_idx_in_features = self.feature_cols.index(self.target_col)
        y = y_unsequenced[:, target_idx_in_features].reshape(-1, 1)
        
        test_size = self.config.get("training_params", {}).get("test_size", 0.2)
        split_idx = int(len(X) * (1 - test_size))
        
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        time_index_test = raw_data.index[split_idx + sequence_length:]

        model = self._create_model(X_train.shape)
        live_predict_cb = LivePredictionCallback(pipeline=self, X_val=X_test, y_val=y_test, time_index_val=time_index_test)
        all_callbacks = [live_predict_cb] + (callbacks or [])
        self.learner = self._create_learner(model, all_callbacks)
        
        epochs = int(self.config.get("training_params", {}).get("epochs", 50))
        history = self.learner.fit(X_train, y_train, epochs=epochs, pipeline_name=self.config.get("pipeline_name"))
        
        final_results = live_predict_cb.last_results or {"history": history, "final_loss": history['loss'][-1] if history.get('loss') else None}
        generate_regression_report(final_results, self.config)
        return final_results
    
    def prepare_data_for_prediction(self, new_data_df: pd.DataFrame) -> np.ndarray:
        """
        Dışarıdan gelen yeni veriyi, yüklenmiş bir modelle tahmin yapmak için
        gerekli formata (ölçeklendirme, sekanslama) sokar.
        """
        if not self.is_fitted:
            raise RuntimeError("Pipeline scalers are not fitted. Call `run(skip_training=True)` on a historical dataset first.")
        self.logger.info("Preparing new data for prediction...")
        
        sequence_length = self.config.get("model_params", {}).get("sequence_length", 60)
        if len(new_data_df) < sequence_length:
            raise ValueError(f"Prediction data must contain at least {sequence_length} rows.")
            
        relevant_data = new_data_df.tail(sequence_length)
        if not all(col in relevant_data.columns for col in self.feature_cols):
            raise ValueError(f"Prediction data is missing required columns: {set(self.feature_cols) - set(relevant_data.columns)}")

        target_series_processed = np.log1p(relevant_data[self.target_col]) if self.config.get("feature_engineering", {}).get("target_col_transform") == 'log' else relevant_data[self.target_col]
        
        scaled_features = self.feature_scaler.transform(relevant_data[self.feature_cols])
        scaled_target = self.scaler.transform(target_series_processed.values.reshape(-1, 1))
        
        scaled_data = np.concatenate([scaled_features, scaled_target], axis=1)
        model_input = scaled_data.reshape(1, sequence_length, scaled_data.shape[1])
        
        self.logger.info(f"Data prepared for prediction with shape: {model_input.shape}")
        return model_input