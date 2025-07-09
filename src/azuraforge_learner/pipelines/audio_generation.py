import os
from abc import abstractmethod
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from pydantic import BaseModel
from scipy.io import wavfile
from importlib import resources
from scipy.signal import resample

from .base import BasePipeline
from ..callbacks import Callback
from ..learner import Learner
from ..losses import CrossEntropyLoss
from ..models import Sequential
from ..optimizers import Adam
from azuraforge_core import Tensor, xp

class AudioGenerationPipeline(BasePipeline):
    # ... (diğer metotlar aynı, değişiklik yok) ...
    def __init__(self, full_config: Dict[str, Any]): super().__init__(full_config); self.learner: Optional[Learner] = None
    def get_config_model(self) -> Optional[type[BaseModel]]: return None
    @abstractmethod
    def _load_data(self) -> np.ndarray: pass
    def _create_sequences(self, waveform: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
        X = np.array([waveform[i : i + seq_length] for i in range(len(waveform) - seq_length)])
        y = np.array([waveform[i + 1 : i + seq_length + 1] for i in range(len(waveform) - seq_length)])
        return X, y
    @abstractmethod
    def _create_model(self, vocab_size: int) -> Sequential: pass
    def run(self, callbacks: Optional[List[Callback]] = None, skip_training: bool = False) -> Dict[str, Any]:
        self.logger.info(f"Running AudioGenerationPipeline: '{self.config.get('pipeline_name')}'..."); encoded_waveform = self._load_data(); vocab_size = int(encoded_waveform.max() + 1); seq_length = self.config.get("model_params", {}).get("sequence_length", 128); X_train, y_train = self._create_sequences(encoded_waveform, seq_length); model = self._create_model(vocab_size); training_params = self.config.get("training_params", {}); lr = float(training_params.get("lr", 0.001)); optimizer = Adam(model.parameters(), lr=lr); self.learner = Learner(model, CrossEntropyLoss(), optimizer, callbacks=callbacks, grad_clip_value=1.0); epochs = int(training_params.get("epochs", 5)); batch_size = int(training_params.get("batch_size", 64)); history = self.learner.fit(X_train, y_train.astype(np.int64), epochs=epochs, batch_size=batch_size, pipeline_name=self.config.get("pipeline_name")); self.logger.info("Generating a sample audio clip after training..."); seed_sequence = X_train[np.random.randint(0, len(X_train))]; sample_rate = self.config.get("data_sourcing", {}).get("sample_rate", 8000); generated_audio_quantized = self.generate(seed_sequence, generation_length=sample_rate * 5); quantization_channels = vocab_size; mu = float(quantization_channels - 1); encoded_float = (generated_audio_quantized.astype(np.float32) / mu) * 2 - 1; decoded_float = np.sign(encoded_float) * (np.expm1(np.abs(encoded_float) * np.log1p(mu))) / mu; generated_audio_16bit = (decoded_float * 32767).astype(np.int16); output_path = None; output_dir = self.config.get("experiment_dir");
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "generated_sample.wav")
            wavfile.write(output_path, sample_rate, generated_audio_16bit)
            self.logger.info(f"Generated audio sample saved to: {output_path}")
        return {"history": history, "generated_audio_path": output_path}     

    def generate(self, initial_seed: np.ndarray, generation_length: int) -> np.ndarray:
        if not self.learner or not self.learner.model:
            raise RuntimeError("Model has not been trained or loaded. Cannot generate audio.")
        
        self.logger.info(f"Starting audio generation for {generation_length} samples...")
        current_sequence_np = initial_seed.copy()
        generated_waveform = []

        self.learner.model.eval()

        for i in range(generation_length):
            if (i + 1) % 1000 == 0:
                self.logger.debug(f"Generating sample {i+1}/{generation_length}...")

            input_sequence = current_sequence_np.reshape(1, -1)
            # predict artık cihaz üzerinde (GPU'da) bir dizi döndürecek
            logits = self.learner.predict(input_sequence) 
            last_step_logits = logits[0, -1, :]
            
            # xp (cupy) fonksiyonlarını doğrudan kullanabiliriz
            probs = xp.exp(last_step_logits) / xp.sum(xp.exp(last_step_logits))
            # Sadece np.random.choice için CPU'ya (numpy) çekiyoruz
            next_sample = np.random.choice(len(probs), p=xp.asnumpy(probs))
            
            generated_waveform.append(next_sample)
            
            # current_sequence her zaman bir NumPy dizisi olarak kalmalı
            current_sequence_np = np.roll(current_sequence_np, -1)
            current_sequence_np[-1] = next_sample
            
        self.logger.info("Audio generation finished.")
        self.learner.model.train()
        return np.array(generated_waveform, dtype=np.uint8)