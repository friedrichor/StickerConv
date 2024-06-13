from src.processor.base_processor import BaseProcessor
from src.processor.blip_processors import (
    BlipCaptionProcessor,
    Blip2ImageTrainProcessor,
    Blip2ImageEvalProcessor
)
from src.common.register import registry


__all__ = [
    "BaseProcessor",
    "BlipCaptionProcessor",
    "Blip2ImageTrainProcessor",
    "Blip2ImageEvalProcessor",
]


def load_processor(name, cfg=None):
    """
    Example
    >>> processor = load_processor("blip2_image_train", cfg=None)
    """
    processor = registry.get_processor_class(name).from_config(cfg)

    return processor