# learner/src/azuraforge_learner/pipelines/__init__.py
"""
Bu __init__ dosyası, `pipelines` paketini bir Python paketi yapar ve
dışa aktarılacak temel pipeline sınıflarını tanımlar.
"""
from .base import BasePipeline
from .timeseries import TimeSeriesPipeline
from .image_classification import ImageClassificationPipeline
from .audio_generation import AudioGenerationPipeline

__all__ = [
    "BasePipeline",
    "TimeSeriesPipeline",
    "ImageClassificationPipeline",
    "AudioGenerationPipeline",
]