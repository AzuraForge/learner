# learner/src/azuraforge_learner/pipelines/base.py
"""
Bu modül, tüm pipeline'ların miras alması gereken temel soyut sınıf olan
BasePipeline'i içerir.
"""
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, TYPE_CHECKING
from pydantic import BaseModel, ValidationError

# Forward-referanslar için, döngüsel importu önler
if TYPE_CHECKING:
    from ..callbacks import Callback
    from ..learner import Learner

class BasePipeline(ABC):
    """
    Tüm pipeline'ların temel sınıfı. Konfigürasyon doğrulama, logger
    oluşturma ve temel pipeline yapısını sağlar.
    """
    def __init__(self, full_config: Dict[str, Any]):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.raw_config = full_config.copy()
        self.config = self._validate_and_prepare_config(full_config)
        self.learner: Optional['Learner'] = None
        self.logger.info(f"Pipeline '{self.config.get('pipeline_name')}' initialized.")

    def _validate_and_prepare_config(self, full_config: Dict[str, Any]) -> Dict[str, Any]:
        """Gelen konfigürasyonu Pydantic modeline göre doğrular."""
        try:
            ConfigModel = self.get_config_model()
            if not ConfigModel:
                self.logger.info("No Pydantic config model provided. Skipping validation.")
                return full_config

            model_fields = ConfigModel.model_fields.keys()
            config_for_validation = {k: v for k, v in full_config.items() if k in model_fields}
            validated_config = ConfigModel(**config_for_validation).model_dump()
            
            final_config = self.raw_config.copy()
            final_config.update(validated_config)
            return final_config
            
        except ValidationError as e:
            self.logger.error(f"Pydantic config validation failed: {e}", exc_info=True)
            error_details = "\n".join([f"  - Field '{err['loc'][0]}': {err['msg']}" for err in e.errors()])
            raise ValueError(f"Invalid configuration for {self.__class__.__name__}:\n{error_details}") from e
        except Exception as e:
            self.logger.error(f"Unexpected error during config validation: {e}", exc_info=True)
            raise e

    @abstractmethod
    def get_config_model(self) -> Optional[type[BaseModel]]:
        """Pipeline konfigürasyonunu doğrulayacak Pydantic modelini döndürmelidir."""
        pass
    
    @abstractmethod
    def run(self, callbacks: Optional[List['Callback']] = None, skip_training: bool = False) -> Dict[str, Any]:
        """Pipeline'ın ana çalışma mantığını içerir."""
        pass