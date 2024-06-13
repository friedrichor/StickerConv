import logging
from typing import Optional
from omegaconf import DictConfig

from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

from src.common.loader import load_json
from src.processor import BaseProcessor
from src.common.register import registry


class BaseDataset(Dataset):
    def __init__(
        self,
        vision_processor = None,
        text_processor = None,
        images_root: Optional[str] = None,
        annotation_path: Optional[str] = None
    ):
        """
        images_root (string): Root directory of images (e.g. coco/images/)
        annotation_path (string): directory to store the annotation file
        """
        self.vision_processor = vision_processor
        self.text_processor = text_processor
        
        self.images_root = images_root
        if annotation_path is not None:
            self.annotation = load_json(annotation_path)

    def __len__(self):
        return len(self.annotation)

    def collate_fn(self, batch):
        return default_collate(batch)


class BaseDatasetBuilder:
    train_dataset_cls, eval_dataset_cls = None, None

    def __init__(self, config: DictConfig):
        super().__init__()

        self.config = config

        self.vision_processors = {"train": BaseProcessor(), "eval": BaseProcessor()}
        self.text_processors = {"train": BaseProcessor(), "eval": BaseProcessor()}

    def build_dataset(self):
        logging.info("Building datasets...")
        dataset = self.build()  # dataset['train'/'val'/'test']

        return dataset

    def build_processors(self):
        vision_processor_config = self.config.get("vision_processor")
        text_processor_config = self.config.get("text_processor")

        if vision_processor_config is not None:
            vision_train_config = vision_processor_config.get("train")
            vision_eval_config = vision_processor_config.get("eval")

            self.vision_processors["train"] = self._build_processor_from_config(vision_train_config)
            self.vision_processors["eval"] = self._build_processor_from_config(vision_eval_config)

        if text_processor_config is not None:
            text_train_config = text_processor_config.get("train")
            text_eval_config = text_processor_config.get("eval")

            self.text_processors["train"] = self._build_processor_from_config(text_train_config)
            self.text_processors["eval"] = self._build_processor_from_config(text_eval_config)
    
    @staticmethod
    def _build_processor_from_config(config):
        return (
            registry.get_processor_class(config.name).from_config(config)
            if config is not None
            else None
        )
    
    def build(self):
        pass
