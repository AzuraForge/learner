# learner/src/azuraforge_learner/pipelines/timeseries.py
import logging
import os
from abc import abstractmethod
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import pandas as pd
from pydantic import BaseModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from .base import BasePipeline
from ..callbacks import Callback
from ..events import Event
from ..learner import Learner
from ..losses import MSELoss
from ..models import Sequential
from ..optimizers import Adam, SGD
from ..reporting import generate_regression_report

def _create_sequences(data: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys).reshape(-1, data.shape[1] if data.ndim > 1 else -1)

class LivePredictionCallback(Callback):
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
        if not (epoch % self.validate_every == 0 or epoch + 1 == total_epochs or epoch == 0):
            return
        if not self.learner:
            self.pipeline.logger.warning("LivePredictionCallback: Learner not set, skipping validation.")
            return
        try:
            y_pred_scaled = self.learner.predict(self.X_val)
            y_test_unscaled, y_pred_unscaled = self.pipeline._inverse_transform_all(self.y_val, y_pred_scaled)
            event.payload['validation_data'] = {
                "x_axis": [d.isoformat() for d in self.time_index_val],
                "y_true": y_test_unscaled.tolist(), "y_pred": y_pred_unscaled.tolist(),
            }
            from sklearn.metrics import r2_score, mean_absolute_error
            self.last_results = {
                "history": self.learner.history,
                "metrics": {'r2_score': float(r2_score(y_test_unscaled, y_pred_unscaled)), 'mae': float(mean_absolute_error(y_test_unscaled, y_pred_unscaled))},
                "final_loss": event.payload.get("loss"),
                "y_true": y_test_unscaled.tolist(), "y_pred": y_pred_unscaled.tolist(),
                "time_index": [d.isoformat() for d in self.time_index_val], "y_label": self.pipeline.target_col
            }
        except Exception as e:
            self.pipeline.logger.error(f"LivePredictionCallback Error: {e}", exc_info=True)


class TimeSeriesPipeline(BasePipeline):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.target_scaler = MinMaxScaler(feature_range=(-1, 1))
        self.feature_scaler = MinMaxScaler(feature_range=(-1, 1))
        self.target_col: Optional[str] = None
        self.feature_cols: Optional[List[str]] = None
        self.is_fitted: bool = False

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
        if optimizer_type == "adam": optimizer = Adam(model.parameters(), lr=lr)
        elif optimizer_type == "sgd": optimizer = SGD(model.parameters(), lr=lr)
        else: raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
        return Learner(model, MSELoss(), optimizer, callbacks=callbacks)

    def _inverse_transform_all(self, y_true_scaled, y_pred_scaled):
        y_true_unscaled = self.target_scaler.inverse_transform(y_true_scaled)
        y_pred_unscaled = self.target_scaler.inverse_transform(y_pred_scaled)
        if self.config.get("feature_engineering", {}).get("target_col_transform") == 'log':
            y_true_unscaled, y_pred_unscaled = np.expm1(y_true_unscaled), np.expm1(y_pred_unscaled)
        return y_true_unscaled.flatten(), y_pred_unscaled.flatten()
        
    def _fit_scalers(self, data: pd.DataFrame):
        if self.is_fitted: return
        self.logger.info("Fitting data scalers...")
        self.target_col, self.feature_cols = self._get_target_and_feature_cols()
        target_series = data[self.target_col].copy()
        if self.config.get("feature_engineering", {}).get("target_col_transform") == 'log':
            target_series = np.log1p(target_series)
        self.target_scaler.fit(target_series.values.reshape(-1, 1))
        self.feature_scaler.fit(data[self.feature_cols])
        self.is_fitted = True
        self.logger.info("Scalers have been fitted.")

    def _prepare_data_for_training(self, data: pd.DataFrame) -> Tuple:
        self.logger.info("Preparing data for training...")
        self._fit_scalers(data)
        
        data_sourcing_config = self.config.get("data_sourcing", {})
        data_limit = data_sourcing_config.get("data_limit")
        if data_limit and isinstance(data_limit, int) and data_limit > 0:
            self.logger.info(f"Applying data limit: Using last {data_limit} rows.")
            data = data.tail(data_limit)

        scaled_features = self.feature_scaler.transform(data[self.feature_cols])
        sequence_length = self.config.get("model_params", {}).get("sequence_length", 60)
        
        X, y = _create_sequences(scaled_features, sequence_length)
        
        try:
            target_col_index = self.feature_cols.index(self.target_col)
            y = y[:, target_col_index] 
        except (ValueError, IndexError):
             raise ValueError(f"Target column '{self.target_col}' not found in feature columns.")
        
        if len(X) == 0: raise ValueError(f"Not enough data to create sequences (need > {sequence_length} rows).")
        test_size = self.config.get("training_params", {}).get("test_size", 0.2)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
        time_index_test = data.index[len(X_train) + sequence_length:]
        
        self.logger.info(f"Data prepared: X_train shape={X_train.shape}, X_test shape={X_test.shape}")
        return X_train, y_train, X_test, y_test, time_index_test

    def run(self, raw_data: pd.DataFrame, callbacks: Optional[List[Callback]] = None, skip_training: bool = False) -> Dict[str, Any]:
        self.logger.info(f"Running TimeSeriesPipeline: '{self.config.get('pipeline_name')}'...")

        if skip_training:
            self._fit_scalers(raw_data)
            return {"status": "skipped", "message": "Training skipped."}
            
        X_train, y_train, X_test, y_test, time_index_test = self._prepare_data_for_training(raw_data)
        model = self._create_model(input_shape=X_train.shape)
        live_predict_cb = LivePredictionCallback(self, X_test, y_test.reshape(-1, 1), time_index_test)
        
        self.learner = self._create_learner(model, [live_predict_cb] + (callbacks or []))
        epochs = int(self.config.get("training_params", {}).get("epochs", 50))
        self.learner.fit(X_train, y_train.reshape(-1, 1), epochs=epochs, pipeline_name=self.config.get("pipeline_name"))
        
        final_results = live_predict_cb.last_results
        if not final_results:
             y_pred_scaled = self.learner.predict(X_test)
             y_test_unscaled, y_pred_unscaled = self._inverse_transform_all(y_test.reshape(-1, 1), y_pred_scaled)
             from sklearn.metrics import r2_score, mean_absolute_error
             final_results = {
                "history": self.learner.history,
                "metrics": {'r2_score': float(r2_score(y_test_unscaled, y_pred_unscaled)), 'mae': float(mean_absolute_error(y_test_unscaled, y_pred_unscaled))},
                "final_loss": self.learner.history.get('loss', [None])[-1]
             }
        
        # === DEĞİŞİKLİK BAŞLANGICI: Sonuçları zenginleştiriyoruz ===
        # Nihai sonuçlar sözlüğüne, tahmin için gereken sütun bilgilerini ekle.
        final_results['target_col'] = self.target_col
        final_results['feature_cols'] = self.feature_cols
        # === DEĞİŞİKLİK SONU ===

        generate_regression_report(final_results, self.config)
        return final_results
    
    def prepare_data_for_prediction(self, new_data_df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted: raise RuntimeError("Scalers not fitted. Call `run(skip_training=True)` first.")
        sequence_length = self.config.get("model_params", {}).get("sequence_length", 60)
        if len(new_data_df) < sequence_length: raise ValueError(f"Prediction data must have at least {sequence_length} rows.")
        
        relevant_data = new_data_df.tail(sequence_length)
        scaled_features = self.feature_scaler.transform(relevant_data[self.feature_cols])
        
        model_input = scaled_features.reshape(1, sequence_length, -1)
        return model_input