import os
from abc import abstractmethod
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from pydantic import BaseModel
from scipy.io import wavfile
from importlib import resources

from .base import BasePipeline
from ..callbacks import Callback
from ..learner import Learner
from ..losses import CrossEntropyLoss
from ..models import Sequential
from ..optimizers import Adam
from azuraforge_core import Tensor, xp

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
        X = np.array([waveform[i : i + seq_length] for i in range(len(waveform) - seq_length)])
        y = np.array([waveform[i + 1 : i + seq_length + 1] for i in range(len(waveform) - seq_length)])
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
        self.learner = Learner(model, CrossEntropyLoss(), optimizer, callbacks=callbacks)

        epochs = int(training_params.get("epochs", 5))
        batch_size = int(training_params.get("batch_size", 64))
        
        history = self.learner.fit(X_train, y_train.astype(np.int64), epochs=epochs, batch_size=batch_size, pipeline_name=self.config.get("pipeline_name"))
        
        self.logger.info("Generating a sample audio clip after training...")
        seed_sequence = X_train[np.random.randint(0, len(X_train))]
        sample_rate = self.config.get("data_sourcing", {}).get("sample_rate", 8000)
        
        generated_audio_quantized = self.generate(seed_sequence, generation_length=sample_rate * 5)
        
        quantization_channels = vocab_size
        mu = float(quantization_channels - 1)
        encoded_float = (generated_audio_quantized.astype(np.float32) / mu) * 2 - 1
        decoded_float = np.sign(encoded_float) * (np.expm1(np.abs(encoded_float) * np.log1p(mu))) / mu
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

        self.learner.model.eval() # Modeli tahmin moduna al

        for _ in range(generation_length):
            # Girdi (N, T) şeklinde olmalı. LSTM bunu (N, T, 1) yapacak.
            input_sequence = current_sequence.reshape(1, -1)
            
            # Predict metodunu kullanıyoruz, o zaten Tensor'e çeviriyor.
            # Modelimiz (LSTM return_sequences=True + Linear) olduğu için çıktı (N, T, vocab_size) olacak.
            logits = self.learner.predict(input_sequence) 
            
            # Sadece son zaman adımının logitlerini al
            last_step_logits = logits[0, -1, :]
            
            # Softmax ile olasılıkları hesapla
            probs = np.exp(last_step_logits) / np.sum(np.exp(last_step_logits))
            next_sample = np.random.choice(len(probs), p=probs)
            
            generated_waveform.append(next_sample)
            
            # Diziyi kaydır ve yeni örneği sona ekle
            current_sequence = np.roll(current_sequence, -1)
            current_sequence[-1] = next_sample
            
        self.logger.info("Audio generation finished.")
        self.learner.model.train() # Modeli tekrar eğitim moduna al
        return np.array(generated_waveform, dtype=np.uint8)
