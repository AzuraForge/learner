from dataclasses import dataclass, field
from typing import Dict, Any, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from .learner import Learner

EventName = Literal["train_begin", "train_end", "epoch_begin", "epoch_end"]

@dataclass
class Event:
    name: EventName
    learner: 'Learner'
    payload: Dict[str, Any] = field(default_factory=dict)
