# learner/src/azuraforge_learner/pipelines/audio_generation.py
"""
Ses üretimi pipeline'ları için temel sınıfı içerir.
"""
import os
from abc import abstractmethod
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from pydantic import BaseModel
from scipy.io import wavfile

from .base import BasePipeline
from ..callbacks import Callback
from ..learner import Learner
from ..losses import CrossEntropyLoss
from ..models import Sequential
from ..optimizers import Adam
from azuraforge_core import Tensor

class AudioGenerationPipeline(BasePipeline):
    """Ses üretimi görevleri için temel pipeline."""
    
    def __init__(self, full_config: Dict[str, Any]):
        super().__init__(full_config)
        self.learner: Optional[Learner] = None

    def get_config_model(self) -> Optional[type[BaseModel]]:
        return None

    @abstractmethod
    def _load_data(self) -> np.ndarray:
        """
        Ham ses verisini (quantize edilmemiş) veya zaten quantize edilmiş
        bir tamsayı dizisini döndürmelidir.
        """
        pass

    def _create_sequences(self, waveform: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Verilen dalga formundan giriş ve hedef dizileri oluşturur."""
        # Girdi (X): her biri seq_length uzunluğunda diziler.
        X = np.array([waveform[i : i + seq_length] for i in range(len(waveform) - seq_length)])
        # Hedef (y): X'teki her diziden bir sonraki gelen örnek.
        y = np.array([waveform[i + seq_length] for i in range(len(waveform) - seq_length)])
        return X, y

    @abstractmethod
    def _create_model(self, vocab_size: int) -> Sequential:
        """
        Modeli oluşturur. vocab_size, quantize edilmiş sesin alabileceği
        farklı değer sayısıdır (örn: 256).
        """
        pass

    def run(self, callbacks: Optional[List[Callback]] = None, skip_training: bool = False) -> Dict[str, Any]:
        self.logger.info(f"Running AudioGenerationPipeline: '{self.config.get('pipeline_name')}'...")
        
        encoded_waveform = self._load_data()
        vocab_size = int(encoded_waveform.max() + 1)
        
        seq_length = self.config.get("model_params", {}).get("sequence_length", 128)
        X_train, y_train = self._create_sequences(encoded_waveform, seq_length)

        model = self._create_model(vocab_size)
        
        training_params = self.config.get("training_params", {})
        lr = float(training_params.get("lr", 0.001))
        optimizer = Adam(model.parameters(), lr=lr)
        # Bu bir sınıflandırma problemidir: Bir sonraki ses örneği (sınıf) ne olacak?
        self.learner = Learner(model, CrossEntropyLoss(), optimizer, callbacks=callbacks)

        epochs = int(training_params.get("epochs", 5))
        batch_size = int(training_params.get("batch_size", 64))
        
        # y_train (hedef) bir tamsayı dizisi olmalı.
        history = self.learner.fit(X_train, y_train.astype(np.int64), epochs=epochs, batch_size=batch_size, pipeline_name=self.config.get("pipeline_name"))
        
        self.logger.info("Generating a sample audio clip after training...")
        seed_sequence = X_train[np.random.randint(0, len(X_train))]
        sample_rate = self.config.get("data_sourcing", {}).get("sample_rate", 8000)
        
        # 5 saniyelik ses üretelim
        generated_audio_quantized = self.generate(seed_sequence, generation_length=sample_rate * 5)
        
        # Mu-law decode (yaklaşık) ve .wav dosyasına yazma
        quantization_channels = vocab_size
        mu = float(quantization_channels - 1)
        # Quantize edilmiş veriyi [-1, 1] aralığına geri getir
        encoded_float = (generated_audio_quantized.astype(np.float32) / mu) * 2 - 1
        # Ters mu-law formülü
        decoded_float = np.sign(encoded_float) * (np.expm1(np.abs(encoded_float) * np.log1p(mu))) / mu
        # 16-bit integer'a çevir
        generated_audio_16bit = (decoded_float * 32767).astype(np.int16)

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
            input_tensor = Tensor(current_sequence.reshape(1, -1, 1)) # LSTM (N, T, D) bekler
            logits = self.learner.predict(input_tensor) # Çıktı (N, vocab_size) olmalı
            
            # Softmax ile olasılıkları hesapla
            probs = np.exp(logits[0]) / np.sum(np.exp(logits[0]))
            next_sample = np.random.choice(len(probs), p=probs)
            
            generated_waveform.append(next_sample)
            
            # Diziyi kaydır ve yeni örneği sona ekle
            current_sequence = np.roll(current_sequence, -1)
            current_sequence[-1] = next_sample
            
        self.logger.info("Audio generation finished.")
        return np.array(generated_waveform, dtype=np.uint8)
