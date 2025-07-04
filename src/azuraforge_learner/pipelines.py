# learner/src/azuraforge_learner/pipelines.py
"""
Bu modül, AzuraForge'daki tüm AI görevleri için temel soyut pipeline
sınıflarını tanımlar. Pipeline'lar, veri yükleme, ön işleme, model oluşturma,
eğitim ve raporlama adımlarını kapsayan standart bir arayüz sağlar.
"""
import logging
import os
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError
from scipy.io import wavfile
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import MinMaxScaler

from .caching import get_cache_filepath, load_from_cache, save_to_cache
from .callbacks import Callback
from .events import Event
from .learner import Learner
from .losses import MSELoss, CrossEntropyLoss
from .models import Sequential
from .optimizers import Adam, SGD
from .reporting import generate_classification_report, generate_regression_report


# --- Yardımcı Fonksiyonlar ve Sınıflar ---

def _create_sequences(data: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """Zaman serisi verisinden girdi (X) ve hedef (y) sekansları oluşturur."""
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys).reshape(-1, data.shape[1] if data.ndim > 1 else -1)


# --- TEMEL PIPELINE SINIFI ---

class BasePipeline(ABC):
    """
    Tüm pipeline'ların miras alması gereken temel sınıf.
    Konfigürasyon doğrulama ve logger oluşturma gibi ortak işlevleri yönetir.
    """
    def __init__(self, full_config: Dict[str, Any]):
        # 1. Logger'ı her şeyden önce oluştur.
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 2. Ham, orijinal konfigürasyonu ileride kullanmak üzere sakla.
        self.raw_config = full_config.copy()
        
        # 3. Konfigürasyonu doğrula ve nihai halini `self.config`'e ata.
        self.config = self._validate_and_prepare_config(full_config)
        
        self.learner: Optional[Learner] = None
        self.logger.info(f"Pipeline '{self.config.get('pipeline_name')}' initialized.")

    def _validate_and_prepare_config(self, full_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Gelen tam konfigürasyonu, pipeline'ın Pydantic modeline göre doğrular ve
        sistemsel anahtarları koruyarak nihai konfigürasyonu döndürür.
        """
        try:
            ConfigModel = self.get_config_model()
            if not ConfigModel:
                self.logger.info("No Pydantic config model provided for this pipeline. Skipping validation.")
                return full_config

            model_fields = ConfigModel.model_fields.keys()
            config_for_validation = {k: v for k, v in full_config.items() if k in model_fields}
            
            validated_config = ConfigModel(**config_for_validation).model_dump()
            
            final_config = self.raw_config.copy()
            final_config.update(validated_config)
            return final_config
            
        except ValidationError as e:
            self.logger.error(f"Pydantic configuration validation failed for {self.__class__.__name__}: {e}", exc_info=True)
            error_details = "\n".join([f"  - Field '{err['loc'][0]}': {err['msg']}" for err in e.errors()])
            raise ValueError(f"Invalid configuration provided:\n{error_details}") from e
        except Exception as e:
            self.logger.error(f"Unexpected error during config validation: {e}", exc_info=True)
            raise e

    @abstractmethod
    def get_config_model(self) -> Optional[type[BaseModel]]:
        """
        Pipeline konfigürasyonunu doğrulayacak Pydantic modelini döndürmelidir.
        Doğrulama istenmiyorsa None dönebilir.
        """
        pass
    
    @abstractmethod
    def run(self, callbacks: Optional[List[Callback]] = None, skip_training: bool = False) -> Dict[str, Any]:
        """Pipeline'ın ana çalışma mantığını içerir."""
        pass


# --- ZAMAN SERİSİ PIPELINE'I ---

class LivePredictionCallback(Callback):
    """Her 'validate_every' epoch'ta tahmin grafiği için veri üreten callback."""
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
        
        # Her 'validate_every' epoch'ta, ilk ve son epoch'ta çalış
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
                "y_true": y_test_unscaled.tolist(),
                "y_pred": y_pred_unscaled.tolist(),
                "x_label": "Tarih", 
                "y_label": self.pipeline.target_col
            }
            
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
                "y_label": self.pipeline.target_col
            }
        except Exception as e:
            self.pipeline.logger.error(f"LivePredictionCallback Error: {e}", exc_info=True)


class TimeSeriesPipeline(BasePipeline):
    """
    Zaman serisi verileriyle çalışan pipeline'lar için temel sınıf.
    Veri önbellekleme, ölçeklendirme ve sekans oluşturma gibi ortak adımları yönetir.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.target_scaler = MinMaxScaler(feature_range=(-1, 1))
        self.feature_scaler = MinMaxScaler(feature_range=(-1, 1))
        self.target_col: Optional[str] = None
        self.feature_cols: Optional[List[str]] = None
        self.is_fitted: bool = False

    @abstractmethod
    def _load_data_from_source(self) -> pd.DataFrame:
        """Pipeline'a özel veri kaynağını (API, dosya vb.) yükler."""
        pass

    def get_caching_params(self) -> Dict[str, Any]:
        """Önbellek anahtarı oluşturmak için kullanılacak konfigürasyon parametrelerini döndürür."""
        return self.config.get("data_sourcing", {})

    @abstractmethod
    def _get_target_and_feature_cols(self) -> Tuple[str, List[str]]:
        """Hedef sütun adını ve özellik sütun adlarının listesini döndürür."""
        pass

    @abstractmethod
    def _create_model(self, input_shape: Tuple) -> Sequential:
        """Pipeline'a özel AI modelini oluşturur."""
        pass

    def _create_learner(self, model: Sequential, callbacks: Optional[List[Callback]]) -> Learner:
        """Verilen model için bir Learner nesnesi oluşturur."""
        training_params = self.config.get("training_params", {})
        lr = float(training_params.get("lr", 0.001))
        optimizer_type = str(training_params.get("optimizer", "adam")).lower()
        
        if optimizer_type == "adam":
            optimizer = Adam(model.parameters(), lr=lr)
        elif optimizer_type == "sgd":
            optimizer = SGD(model.parameters(), lr=lr)
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
            
        return Learner(model, MSELoss(), optimizer, callbacks=callbacks)

    def _inverse_transform_all(self, y_true_scaled, y_pred_scaled):
        """Tahminleri ve gerçek değerleri orijinal ölçeğine geri döndürür."""
        y_true_unscaled = self.target_scaler.inverse_transform(y_true_scaled)
        y_pred_unscaled = self.target_scaler.inverse_transform(y_pred_scaled)
        
        if self.config.get("feature_engineering", {}).get("target_col_transform") == 'log':
            y_true_unscaled = np.expm1(y_true_unscaled)
            y_pred_unscaled = np.expm1(y_pred_unscaled)
            
        return y_true_unscaled.flatten(), y_pred_unscaled.flatten()
        
    def _fit_scalers(self, data: pd.DataFrame):
        """Verilen veriye göre hedef ve özellik scaler'larını eğitir."""
        if self.is_fitted:
            return
        self.logger.info("Fitting data scalers...")
        self.target_col, self.feature_cols = self._get_target_and_feature_cols()
        
        target_series = data[self.target_col].copy()
        if self.config.get("feature_engineering", {}).get("target_col_transform") == 'log':
            target_series = np.log1p(target_series)
            
        self.target_scaler.fit(target_series.values.reshape(-1, 1))
        self.feature_scaler.fit(data[self.feature_cols])
        self.is_fitted = True
        self.logger.info("Scalers have been fitted.")

    def _load_and_cache_data(self) -> pd.DataFrame:
        """Veriyi önbellekten veya kaynaktan yükler."""
        system_config = self.config.get("system", {})
        if not system_config.get("caching_enabled", True):
            self.logger.info("Caching is disabled. Loading data directly from source.")
            return self._load_data_from_source()

        cache_dir = os.getenv("CACHE_DIR", ".cache")
        cache_filepath = get_cache_filepath(
            cache_dir, 
            self.config.get('pipeline_name', 'default'), 
            self.get_caching_params()
        )
        
        cached_data = load_from_cache(cache_filepath, system_config.get("cache_max_age_hours", 24))
        if cached_data is not None:
            return cached_data
            
        self.logger.info("No valid cache found. Loading data from source.")
        source_data = self._load_data_from_source()
        if isinstance(source_data, pd.DataFrame) and not source_data.empty:
            save_to_cache(source_data, cache_filepath)
            
        return source_data
        
    def _prepare_data_for_training(self, data: pd.DataFrame) -> Tuple:
        """Veriyi ölçekler, sekanslar ve train/test setlerine ayırır."""
        self.logger.info("Preparing data for training...")
        self._fit_scalers(data)

        target_series_processed = data[self.target_col].copy()
        if self.config.get("feature_engineering", {}).get("target_col_transform") == 'log':
            target_series_processed = np.log1p(target_series_processed)

        scaled_target = self.target_scaler.transform(target_series_processed.values.reshape(-1, 1))
        
        # Sadece hedef sütun ile sekans oluşturma (tek değişkenli)
        # Çok değişkenli modeller için bu kısım genişletilebilir.
        sequence_length = self.config.get("model_params", {}).get("sequence_length", 60)
        X, y = _create_sequences(scaled_target, sequence_length)
        if len(X) == 0:
            raise ValueError(f"Not enough data to create sequences. Need > {sequence_length} rows, but got {len(data)}.")
        
        test_size = self.config.get("training_params", {}).get("test_size", 0.2)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
        
        time_index_test = data.index[len(X_train) + sequence_length:]

        self.logger.info(f"Data prepared: X_train shape={X_train.shape}, X_test shape={X_test.shape}")
        return X_train, y_train, X_test, y_test, time_index_test

    def run(self, callbacks: Optional[List[Callback]] = None, skip_training: bool = False) -> Dict[str, Any]:
        """Zaman serisi pipeline'ının ana çalışma döngüsü."""
        self.logger.info(f"Running TimeSeriesPipeline: '{self.config.get('pipeline_name')}'...")
        
        raw_data = self._load_and_cache_data()

        if skip_training:
            self._fit_scalers(raw_data)
            self.logger.info("`skip_training` is True. Pipeline run finished after data preparation.")
            return {"status": "skipped", "message": "Training was skipped."}

        X_train, y_train, X_test, y_test, time_index_test = self._prepare_data_for_training(raw_data)
        
        # Model girdisi (batch, seq_len, features) şeklinde olmalı
        model = self._create_model(input_shape=X_train.shape)
        
        live_predict_cb = LivePredictionCallback(
            pipeline=self, X_val=X_test, y_val=y_test, time_index_val=time_index_test
        )
        all_callbacks = [live_predict_cb] + (callbacks or [])
        
        self.learner = self._create_learner(model, all_callbacks)
        
        epochs = int(self.config.get("training_params", {}).get("epochs", 50))
        self.learner.fit(X_train, y_train, epochs=epochs, pipeline_name=self.config.get("pipeline_name"))
        
        final_results = live_predict_cb.last_results
        if not final_results:
             y_pred_scaled = self.learner.predict(X_test)
             y_test_unscaled, y_pred_unscaled = self._inverse_transform_all(y_test, y_pred_scaled)
             from sklearn.metrics import r2_score, mean_absolute_error
             final_results = {
                "history": self.learner.history,
                "metrics": {'r2_score': float(r2_score(y_test_unscaled, y_pred_unscaled)), 'mae': float(mean_absolute_error(y_test_unscaled, y_pred_unscaled))},
                "final_loss": self.learner.history.get('loss', [None])[-1]
             }
             
        generate_regression_report(final_results, self.config)
        return final_results
    
    def prepare_data_for_prediction(self, new_data_df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Pipeline scalers are not fitted. Call `run(skip_training=True)` on a historical dataset first.")
        self.logger.info("Preparing new data for prediction...")
        
        sequence_length = self.config.get("model_params", {}).get("sequence_length", 60)
        if len(new_data_df) < sequence_length:
            raise ValueError(f"Prediction data must contain at least {sequence_length} rows.")
            
        relevant_data = new_data_df.tail(sequence_length)
        
        target_series_processed = relevant_data[self.target_col].copy()
        if self.config.get("feature_engineering", {}).get("target_col_transform") == 'log':
            target_series_processed = np.log1p(target_series_processed)
        
        scaled_target = self.target_scaler.transform(target_series_processed.values.reshape(-1, 1))
        
        model_input = scaled_target.reshape(1, sequence_length, -1)
        
        self.logger.info(f"Data prepared for prediction with shape: {model_input.shape}")
        return model_input

# --- GÖRÜNTÜ SINIFLANDIRMA PIPELINE'I ---

class ImageClassificationPipeline(BasePipeline):
    """Görüntü sınıflandırma görevleri için temel pipeline."""
    
    def get_config_model(self) -> Optional[type[BaseModel]]:
        return None # Şimdilik bu pipeline için Pydantic doğrulaması yok

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


# --- SES ÜRETİM PIPELINE'I ---

class AudioGenerationPipeline(BasePipeline):
    """Ses üretimi görevleri için temel pipeline."""

    def get_config_model(self) -> Optional[type[BaseModel]]:
        return None

    @abstractmethod
    def _load_data(self) -> np.ndarray:
        pass

    def _create_sequences(self, waveform: np.ndarray, seq_length: int):
        X = [waveform[i:i+seq_length] for i in range(len(waveform) - seq_length -1)]
        y = [waveform[i+1:i+seq_length+1] for i in range(len(waveform) - seq_length -1)]
        return np.array(X), np.array(y)

    @abstractmethod
    def _create_model(self, vocab_size: int) -> Sequential:
        pass

    def run(self, callbacks: Optional[List[Callback]] = None, skip_training: bool = False) -> Dict[str, Any]:
        self.logger.info(f"Running AudioGenerationPipeline: '{self.config.get('pipeline_name')}'...")
        
        waveform = self._load_data()
        vocab_size = int(waveform.max() + 1)
        
        seq_length = self.config.get("model_params", {}).get("sequence_length", 128)
        X_train, y_train = self._create_sequences(waveform, seq_length)

        model = self._create_model(vocab_size)
        
        training_params = self.config.get("training_params", {})
        lr = float(training_params.get("lr", 0.001))
        optimizer = Adam(model.parameters(), lr=lr)
        self.learner = Learner(model, CrossEntropyLoss(), optimizer, callbacks=callbacks)

        epochs = int(training_params.get("epochs", 5))
        history = self.learner.fit(X_train, y_train.astype(np.int32), epochs=epochs, pipeline_name=self.config.get("pipeline_name"))
        
        self.logger.info("Generating a sample audio clip after training...")
        seed_sequence = X_train[np.random.randint(0, len(X_train))]
        
        sample_rate = self.config.get("data_sourcing", {}).get("sample_rate", 8000)
        generated_audio_quantized = self.generate(seed_sequence, generation_length=sample_rate * 5)
        
        generated_audio_float = (generated_audio_quantized / 255.0 * 2.0) - 1.0
        generated_audio_16bit = (generated_audio_float * 32767).astype(np.int16)
        
        output_path = None
        output_dir = self.config.get("experiment_dir")
        if output_dir:
            output_path = os.path.join(output_dir, "generated_sample.wav")
            wavfile.write(output_path, sample_rate, generated_audio_16bit)
            self.logger.info(f"Generated audio sample saved to: {output_path}")
        
        return {"history": history, "generated_audio_path": output_path}     

    def generate(self, initial_seed: np.ndarray, generation_length: int) -> np.ndarray:
        if not self.learner or not self.learner.model:
            raise RuntimeError("Model has not been trained or loaded. Cannot generate audio.")
        
        self.logger.info(f"Starting audio generation for {generation_length} samples...")
        
        current_sequence = initial_seed.copy()
        generated_waveform = []

        for _ in range(generation_length):
            input_tensor = Tensor(current_sequence.reshape(1, -1))
            logits = self.learner.predict(input_tensor) 
            last_step_logits = logits[0, -1, :]
            
            probs = np.exp(last_step_logits) / np.sum(np.exp(last_step_logits))
            
            next_sample = np.random.choice(len(probs), p=probs)
            
            generated_waveform.append(next_sample)
            
            current_sequence = np.roll(current_sequence, -1)
            current_sequence[-1] = next_sample
            
        self.logger.info("Audio generation finished.")
        return np.array(generated_waveform, dtype=np.uint8)