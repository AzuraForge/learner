# learner/src/azuraforge_learner/pipelines/audio_generation.py
"""
Ses üretimi pipeline'ları için temel sınıfı içerir.
"""
import os
# DÜZELTME: Gerekli tüm importlar eklendi
from abc import abstractmethod
from typing import Dict, Any, Optional, Tuple, List

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
    def get_config_model(self) -> Optional[type[BaseModel]]:
        return None

    @abstractmethod
    def _load_data(self) -> np.ndarray:
        pass

    def _create_sequences(self, waveform: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
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