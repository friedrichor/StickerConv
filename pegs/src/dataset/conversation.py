import os
from PIL import Image
import torch

from src.dataset.base import BaseDataset, BaseDatasetBuilder
from src.common.register import registry


class ConversationDataset(BaseDataset):
    def __init__(self, vision_processor, text_processor, images_root, annotation_path):
        super().__init__(vision_processor, text_processor, images_root, annotation_path)
        
    def __getitem__(self, index):
        data = self.annotation[index]
        
        pixel_values_list = []
        for image in data["image"]:
            image_path = os.path.join(self.images_root, image)
            image = Image.open(image_path).convert("RGB")
            pixel_values = self.vision_processor(image)
            pixel_values_list.append(pixel_values)
        stacked_pixel_values = torch.stack(pixel_values_list)
        
        caption = data["caption"]
        conversation = data["conversation"]
        
        return {
            "pixel_values": stacked_pixel_values,
            "caption": caption,
            "conversation": conversation,
        }
    
    def collate_fn(self, batch):
        pixel_values = [instance["pixel_values"] for instance in batch]
        caption = [instance["caption"] for instance in batch]
        conversation = [instance["conversation"] for instance in batch]
        
        return {
            "pixel_values": pixel_values,
            "caption": caption,
            "conversation": conversation
        }


@registry.register_builder("conversation")
class ConversationDatasetBuilder(BaseDatasetBuilder):
    train_dataset_cls = ConversationDataset

    def build(self):
        self.build_processors()
        
        storage = self.config.storage
        images_root = storage.images
        annotation_path = storage.annotation

        datasets = dict()
        split = "train"

        if not os.path.exists(images_root):
            raise ValueError("storage path {} does not exist.".format(images_root))
        if not os.path.exists(annotation_path):
            raise ValueError("storage path {} does not exist.".format(annotation_path))

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets[split] = dataset_cls(
            vision_processor=self.vision_processors[split],
            text_processor=self.text_processors[split],
            images_root=images_root,
            annotation_path=annotation_path
        )

        return datasets
