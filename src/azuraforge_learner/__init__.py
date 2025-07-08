# learner/src/azuraforge_learner/__init__.py
"""
AzuraForge Learner kütüphanesinin ana paketi.
Yüksek seviyeli AI bileşenlerini dışa aktarır.
"""
from .events import Event
from .callbacks import Callback
from .losses import Loss, MSELoss, CrossEntropyLoss
# --- DEĞİŞİKLİK BURADA ---
# Artık doğrudan yeni `layers` alt paketinden import ediyoruz.
from .layers import (
    Layer, Linear, ReLU, Sigmoid, LSTM, Conv2D, 
    MaxPool2D, Flatten, Embedding, Attention, Dropout
)
# --- DEĞİŞİKLİK SONU ---
from .models import Sequential
from .optimizers import Optimizer, SGD, Adam
from .learner import Learner
from .pipelines import (
    BasePipeline, 
    TimeSeriesPipeline, 
    ImageClassificationPipeline, 
    AudioGenerationPipeline
)


__all__ = [
    "Event", "Callback",
    "Loss", "MSELoss", "CrossEntropyLoss",
    "Layer", "Linear", "ReLU", "Sigmoid", "LSTM", "Conv2D", "MaxPool2D", 
    "Flatten", "Embedding", "Attention", "Dropout",
    "Sequential", 
    "Optimizer", "SGD", "Adam",
    "Learner",
    "BasePipeline", 
    "TimeSeriesPipeline",
    "ImageClassificationPipeline",
    "AudioGenerationPipeline"
]
