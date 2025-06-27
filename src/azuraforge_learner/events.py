# ========== YENİ DOSYA: src/azuraforge_learner/events.py ==========
from dataclasses import dataclass, field
from typing import Dict, Any, Literal

# Olay adlarını standartlaştıralım
EventName = Literal[
    "train_begin", "train_end",
    "epoch_begin", "epoch_end",
    "batch_begin", "batch_end"
]

@dataclass
class Event:
    """
    Eğitim döngüsü içinde taşınan standart olay nesnesi.
    """
    name: EventName
    learner: 'Learner' # Döngüsel importu önlemek için string olarak tip tanımı
    payload: Dict[str, Any] = field(default_factory=dict)