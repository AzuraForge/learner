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
from azuraforge_core import Tensor # Tensor'u import ediyoruz

def _create_sequences(data: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    # y için target_col_index kullanmak yerine, TimeSeriesPipeline'ın kendisi
    # scaler ile transform ederken sadece hedef sütunu alacak.
    # _create_sequences'ın görevi sadece sıraları oluşturmak.
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
            
            # === DÜZELTME: Live tahminde x_axis ve y_true/y_pred boyut uyumsuzluğunu gider ===
            # TimeSeriesPipeline'ın kendisi scalerları kullanarak son N noktayı döndürdüğünden,
            # burada da uygun şekilde eşleşmeli.
            # Normalde validation_data'nın X_val ile uyumlu olması gerekir.
            # Şimdilik, sadece `predict`ten gelen `y_true` ve `y_pred`'i kullanıyoruz.
            
            event.payload['validation_data'] = {
                "x_axis": [d.isoformat() for d in self.time_index_val],
                "y_true": y_test_unscaled.tolist(), 
                "y_pred": y_pred_unscaled.tolist(),
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
        # Scaler'lar pipeline örneği başına saklanır.
        self.target_scaler: MinMaxScaler = MinMaxScaler(feature_range=(-1, 1))
        self.feature_scaler: MinMaxScaler = MinMaxScaler(feature_range=(-1, 1))
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
        # ÖLÇEKLEME: y_true_scaled ve y_pred_scaled'i 2 boyutlu hale getiriyoruz.
        y_true_scaled = y_true_scaled.reshape(-1, 1)
        y_pred_scaled = y_pred_scaled.reshape(-1, 1)

        y_true_unscaled = self.target_scaler.inverse_transform(y_true_scaled)
        y_pred_unscaled = self.target_scaler.inverse_transform(y_pred_scaled)
        
        if self.config.get("feature_engineering", {}).get("target_col_transform") == 'log':
            # log1p'nin tersi expm1'dir
            y_true_unscaled = np.expm1(y_true_unscaled)
            y_pred_unscaled = np.expm1(y_pred_unscaled)
        
        return y_true_unscaled.flatten(), y_pred_unscaled.flatten()
        
    def _fit_scalers(self, data: pd.DataFrame):
        # Bu metod sadece `run` veya `predict` çağrıldığında ilk kez çalışır.
        if self.is_fitted:
            self.logger.info("Scalers already fitted. Skipping.")
            return
            
        self.logger.info("Fitting data scalers...")
        self.target_col, self.feature_cols = self._get_target_and_feature_cols()
        
        # Hedef sütunu ölçeklemeden önce bir kopyasını al
        target_series = data[self.target_col].copy()
        if self.config.get("feature_engineering", {}).get("target_col_transform") == 'log':
            target_series = np.log1p(target_series)
        
        # target_scaler sadece hedef sütun üzerinde eğitilir
        self.target_scaler.fit(target_series.values.reshape(-1, 1))

        # feature_scaler tüm özellik sütunları üzerinde eğitilir
        self.feature_scaler.fit(data[self.feature_cols])
        
        self.is_fitted = True
        self.logger.info("Scalers have been fitted.")

    def _prepare_data_for_training(self, data: pd.DataFrame) -> Tuple:
        self.logger.info("Preparing data for training...")
        self._fit_scalers(data) # Scaler'ların eğitildiğinden emin ol
        
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
            # y'nin sadece hedef sütuna ait değerleri içerdiğinden emin ol
            y_target_only = y[:, target_col_index].reshape(-1, 1) # 2 boyutlu olmalı
        except (ValueError, IndexError):
             raise ValueError(f"Target column '{self.target_col}' not found in feature columns.")
        
        if len(X) == 0: raise ValueError(f"Not enough data to create sequences (need > {sequence_length} rows).")
        test_size = self.config.get("training_params", {}).get("test_size", 0.2)
        X_train, X_test, y_train, y_test = train_test_split(X, y_target_only, test_size=test_size, shuffle=False)
        
        # time_index_test, X_test'in sonuna denk gelen gerçek zaman indeksidir.
        # Bu, prediction_comparison grafiği için kullanılır.
        time_index_for_sequences = data.index[sequence_length:]
        time_index_test = time_index_for_sequences[len(X_train):] # X_test'e karşılık gelen zaman dilimi
        
        self.logger.info(f"Data prepared: X_train shape={X_train.shape}, X_test shape={X_test.shape}")
        return X_train, y_train, X_test, y_test, time_index_test

    def run(self, raw_data: pd.DataFrame, callbacks: Optional[List[Callback]] = None, skip_training: bool = False) -> Dict[str, Any]:
        self.logger.info(f"Running TimeSeriesPipeline: '{self.config.get('pipeline_name')}'...")

        self._fit_scalers(raw_data) # Scaler'ları başta bir kez eğit.

        if skip_training:
            self.logger.info("Training skipped as requested.")
            return {"status": "skipped", "message": "Training skipped."}
            
        X_train, y_train, X_test, y_test, time_index_test = self._prepare_data_for_training(raw_data)
        
        # Modeli oluştururken input_shape'in 3 boyutlu olduğundan emin olmalıyız.
        # X_train.shape (batch_size, sequence_length, num_features)
        model = self._create_model(input_shape=X_train.shape) 
        
        live_predict_cb = LivePredictionCallback(self, X_test, y_test, time_index_test)
        
        self.learner = self._create_learner(model, [live_predict_cb] + (callbacks or []))
        epochs = int(self.config.get("training_params", {}).get("epochs", 50))
        self.learner.fit(X_train, y_train, epochs=epochs, pipeline_name=self.config.get("pipeline_name"))
        
        final_results = live_predict_cb.last_results
        if not final_results:
             # Eğer callback çalışmadıysa veya hata verdiyse manuel olarak sonuçları al
             y_pred_scaled = self.learner.predict(X_test)
             y_test_unscaled, y_pred_unscaled = self._inverse_transform_all(y_test, y_pred_scaled)
             from sklearn.metrics import r2_score, mean_absolute_error
             final_results = {
                "history": self.learner.history,
                "metrics": {'r2_score': float(r2_score(y_test_unscaled, y_pred_unscaled)), 'mae': float(mean_absolute_error(y_test_unscaled, y_pred_unscaled))},
                "final_loss": self.learner.history.get('loss', [None])[-1],
                "y_true": y_test_unscaled.tolist(), 
                "y_pred": y_pred_unscaled.tolist(),
                "time_index": [d.isoformat() for d in time_index_test], 
                "y_label": self.target_col
             }
        
        # Nihai sonuçlar sözlüğüne, tahmin için gereken sütun bilgilerini ekle.
        final_results['target_col'] = self.target_col
        final_results['feature_cols'] = self.feature_cols

        generate_regression_report(final_results, self.config)
        return final_results
    
    def prepare_data_for_prediction(self, new_data_df: pd.DataFrame) -> np.ndarray:
        # Yeni verinin son sequence_length kadarını al
        sequence_length = self.config.get("model_params", {}).get("sequence_length", 60)
        
        # Eğer yeni veri yeterli değilse hata ver
        if len(new_data_df) < sequence_length: 
            raise ValueError(f"Prediction data must have at least {sequence_length} rows.")
        
        # Sadece modelin beklediği feature_cols'ları kullan
        relevant_data_for_features = new_data_df.tail(sequence_length)[self.feature_cols]
        
        # Özellikleri ölçekle
        scaled_features = self.feature_scaler.transform(relevant_data_for_features)
        
        # Modela uygun 3 boyutlu şekle getir (batch_size=1, sequence_length, num_features)
        model_input = scaled_features.reshape(1, sequence_length, -1)
        return model_input

    # === YENİ FONKSİYON: Çok Adımlı Tahmin ===
    def forecast(self, initial_df: pd.DataFrame, learner: Learner, num_steps: int) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Scalers not fitted. Call `run(skip_training=True)` or `_fit_scalers` first.")
            
        self.logger.info(f"Generating {num_steps} future predictions...")
        
        # Tahmin için son sequence_length verisini al
        sequence_length = self.config.get("model_params", {}).get("sequence_length", 60)
        
        # İlk tahmin için kullanılacak veri, `initial_df`'in son `sequence_length` adımı olmalı
        current_sequence_df = initial_df.tail(sequence_length).copy()

        forecast_data = []
        last_index = current_sequence_df.index[-1]
        
        # Zaman aralığını otomatik belirle (saatlik veya günlük)
        time_diff = pd.Timedelta(current_sequence_df.index[1] - current_sequence_df.index[0])
        self.logger.info(f"Detected time interval: {time_diff}")

        for i in range(num_steps):
            # Model girdisini hazırla
            prepared_data = self.prepare_data_for_prediction(current_sequence_df)
            
            # Tahmin yap
            scaled_prediction_value = learner.predict(prepared_data)
            
            # Tahmini ters ölçekle ve ters log dönüşümü yap
            unscaled_prediction_value_flat, _ = self._inverse_transform_all(scaled_prediction_value, scaled_prediction_value)
            
            predicted_value = unscaled_prediction_value_flat[0]
            
            # Bir sonraki zaman adımını oluştur
            next_index = last_index + time_diff * (i + 1)
            
            # Tahmin verisini kaydet
            forecast_data.append({'time': next_index, 'predicted_value': predicted_value})
            
            # Bir sonraki iterasyon için current_sequence_df'i güncelle
            # Yeni bir DataFrame satırı oluştur
            new_row_dict = {col: 0.0 for col in self.feature_cols} # Varsayılan olarak 0, sadece tahmin edilen hedef sütun güncellenecek
            
            # Hedef sütununun indeksini bul ve tahmin edilen değerle güncelle
            if self.target_col in new_row_dict:
                new_row_dict[self.target_col] = predicted_value
            
            # Yeni satırı DataFrame'e ekle ve eski ilk satırı çıkar
            new_row_df = pd.DataFrame([new_row_dict], index=[next_index])
            current_sequence_df = pd.concat([current_sequence_df, new_row_df]).tail(sequence_length)
            
            # Yeni current_sequence_df'i ölçekleyip modele verebilmek için
            # target_col dışındaki diğer feature_cols için değerlere ihtiyacımız var.
            # Şu an sadece target_col'u tahminle dolduruyoruz, diğerleri 0 kalıyor.
            # Gerçek uygulamalarda bu diğer feature'lar için de bir tahmin mekanizması
            # veya gelecekteki bilinen değerler (örn: hava durumu için rüzgar hızı) gerekebilir.
            # Şimdilik basitleştirilmiş bir yaklaşım izliyoruz.
            
            # Not: current_sequence_df artık ham değerler içeriyor, scaled değil.
            # prepare_data_for_prediction methodu bunu otomatik olarak ölçekleyecek.
            
        forecast_df = pd.DataFrame(forecast_data).set_index('time')
        self.logger.info(f"Finished generating {num_steps} future predictions.")
        return forecast_df