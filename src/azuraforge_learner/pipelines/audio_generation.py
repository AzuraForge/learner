import os
from abc import abstractmethod
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from pydantic import BaseModel
from scipy.io import wavfile
from scipy import signal
from importlib import resources

from .base import BasePipeline
from ..callbacks import Callback
from ..learner import Learner
from ..losses import CrossEntropyLoss
from ..models import Sequential
from ..optimizers import Adam
from azuraforge_core import Tensor, xp, DEVICE

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
        """Verilen dalga formundan giriş (X) ve hedef (y) dizileri oluşturur."""
        # Girdi (X): her biri seq_length uzunluğunda diziler.
        # Örnek: waveform = [0,1,2,3,4], seq_length = 3
        # X = [[0,1,2], [1,2,3]]
        X = np.array([waveform[i : i + seq_length] for i in range(len(waveform) - seq_length - 1)])
        
        # === KRİTİK DÜZELTME ===
        # Hedef (y): Girdinin bir zaman adımı kaydırılmış hali olmalıdır.
        # Model, t anındaki girdiyi alıp t+1 anındaki çıktıyı tahmin etmeyi öğrenir.
        # Örnek: waveform = [0,1,2,3,4], seq_length = 3
        # y = [[1,2,3], [2,3,4]]
        y = np.array([waveform[i + 1 : i + seq_length + 1] for i in range(len(waveform) - seq_length - 1)])
        # === DÜZELTME SONU ===
        
        return X, y

    @abstractmethod
    def _create_model(self, vocab_size: int) -> Sequential:
        """
        Modeli oluşturur. vocab_size, quantize edilmiş sesin alabileceği
        farklı değer sayısıdır (örn: 256).
        """
        pass

    def run(self, callbacks: Optional[List[Callback]] = None, skip_training: bool = False) -> Dict[str, Any]:
        # === YENİ: experiment_dir oluşturma adımı ===
        output_dir = self.config.get("experiment_dir")
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            self.logger.info(f"Ensured experiment directory exists: {output_dir}")
        # === BİTTİ ===

        self.logger.info(f"Running AudioGenerationPipeline: '{self.config.get('pipeline_name')}'...")
        
        encoded_waveform = self._load_data()
        vocab_size = int(encoded_waveform.max() + 1)
        
        seq_length = self.config.get("model_params", {}).get("sequence_length", 128)
        X_train, y_train = self._create_sequences(encoded_waveform, seq_length)

        model = self._create_model(vocab_size)
        
        training_params = self.config.get("training_params", {})
        lr = float(training_params.get("lr", 0.001))
        optimizer = Adam(model.parameters(), lr=lr)

        self.learner = Learner(
            model, 
            CrossEntropyLoss(), 
            optimizer, 
            callbacks=callbacks,
            grad_clip_value=1.0
        )

        epochs = int(training_params.get("epochs", 10))
        batch_size = int(training_params.get("batch_size", 128))
        
        history = self.learner.fit(X_train, y_train.astype(np.int64), epochs=epochs, batch_size=batch_size, pipeline_name=self.config.get("pipeline_name"))
        
        self.logger.info("Generating a sample audio clip after training...")
        start_index = np.random.randint(0, len(X_train) - 1)
        seed_sequence = X_train[start_index]
        sample_rate = self.config.get("data_sourcing", {}).get("sample_rate", 8000)
        
        generated_audio_quantized = self.generate(seed_sequence, generation_length=sample_rate * 5)
        
        quantization_channels = vocab_size
        mu = float(quantization_channels - 1)
        encoded_float = (generated_audio_quantized.astype(np.float32) / mu) * 2 - 1
        decoded_float = np.sign(encoded_float) * (np.expm1(np.abs(encoded_float) * np.log1p(mu))) / mu
        generated_audio_16bit = (decoded_float * 32767).astype(np.int16)

        output_path = None
        if output_dir:
            output_path = os.path.join(output_dir, "generated_sample.wav")
            wavfile.write(output_path, sample_rate, generated_audio_16bit)
            self.logger.info(f"Generated audio sample saved to: {output_path}")
        
        return {"history": history, "generated_audio_path": output_path}     

    def generate(self, initial_seed: np.ndarray, generation_length: int) -> np.ndarray:
        if not self.learner or not self.learner.model:
            raise RuntimeError("Model has not been trained or loaded. Cannot generate audio.")
        
        self.logger.info(f"Starting audio generation for {generation_length} samples...")
        
        self.learner.model.eval()

        # === PERFORMANS OPTİMİZASYONU ===
        # Başlangıç dizisini en başta GPU'ya (veya mevcut cihaza) gönder.
        current_sequence_gpu = xp.asarray(initial_seed)
        generated_waveform_gpu = xp.zeros(generation_length, dtype=xp.uint8)

        for i in range(generation_length):
            # Girdi artık her zaman doğru cihazda ve 2D (1, T) şeklinde.
            input_tensor = Tensor(current_sequence_gpu.reshape(1, -1))
            
            # `predict` artık bir NumPy dizisi döndürüyor, tekrar Tensor'e çevirmeye gerek yok.
            logits = self.learner.model(input_tensor) # Doğrudan model.forward çağrısı
            
            # Hesaplamaları GPU üzerinde (xp ile) yap
            last_step_logits = logits.data[0, -1, :]
            probs = xp.exp(last_step_logits) / xp.sum(xp.exp(last_step_logits))
            
            # `xp.random.choice` CuPy'de yok, bu yüzden olasılıkları CPU'ya çekip NumPy kullanmalıyız.
            # Ancak bu, her adımda senkronizasyon gerektirir. Daha verimli bir yol arayalım.
            # Şimdilik en basit ve doğru yol, sadece bu kısmı CPU'da yapmaktır.
            if DEVICE == 'gpu':
                probs_cpu = probs.get()
            else:
                probs_cpu = probs
                
            next_sample = np.random.choice(len(probs_cpu), p=probs_cpu)
            
            generated_waveform_gpu[i] = next_sample
            
            # Diziyi GPU üzerinde (xp ile) kaydır
            current_sequence_gpu = xp.roll(current_sequence_gpu, -1)
            current_sequence_gpu[-1] = next_sample

        self.logger.info("Audio generation finished.")
        self.learner.model.train()

        # Sonucu tek seferde CPU'ya çek (eğer GPU'daysa)
        if DEVICE == 'gpu':
            return generated_waveform_gpu.get()
        else:
            return generated_waveform_gpu
        # === OPTİMİZASYON SONU ===