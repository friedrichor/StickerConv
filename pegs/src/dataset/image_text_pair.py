import os
from PIL import Image
import webdataset as wds

from src.dataset.base import BaseDataset, BaseDatasetBuilder
from src.common.register import registry


"""
General
"""
class GeneralImageTextDataset(BaseDataset):
    def __init__(self, vision_processor, text_processor, images_root, annotation_path):
        super().__init__(vision_processor, text_processor, images_root, annotation_path)
        
    def __getitem__(self, index):
        data = self.annotation[index]
        
        image = data["image"]
        caption = data["caption"]
        
        image_path = os.path.join(self.images_root, image)
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.vision_processor(image)
        
        return {
            "pixel_values": pixel_values,
            "caption": caption
        }
        
        
@registry.register_builder("general_image_text")
class GeneralImageTextDatasetBuilder(BaseDatasetBuilder):
    train_dataset_cls = GeneralImageTextDataset

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


"""
Laion (webdataset)
"""
class LaionDataset(BaseDataset):
    def __init__(self, vision_processor, text_processor, storage_path):
        super().__init__(vision_processor=vision_processor, text_processor=text_processor)
        
        self.inner_dataset = wds.DataPipeline(
            wds.ResampledShards(urls=storage_path),
            wds.tarfile_to_samples(handler=wds.warn_and_continue),
            wds.shuffle(1000, handler=wds.warn_and_continue),
            wds.decode("pilrgb", handler=wds.warn_and_continue),
            wds.to_tuple("jpg", "json", handler=wds.warn_and_continue),
            wds.map_tuple(self.vision_processor, handler=wds.warn_and_continue),
            wds.map(self.to_dict, handler=wds.warn_and_continue),
        )

    def to_dict(self, sample):
        return {
            "pixel_values": sample[0],
            "caption": self.text_processor(sample[1]["caption"]),
        }


@registry.register_builder("laion")
class LaionDatasetBuilder(BaseDatasetBuilder):
    train_dataset_class = LaionDataset

    def build(self):
        self.build_processors()

        storage = self.config.storage
        
        dataset = dict()
        split = "train"

        # create datasets
        # [NOTE] return inner_datasets (wds.DataPipeline)
        dataset_class = self.train_dataset_class
        dataset[split] = dataset_class(
            vision_processor=self.vision_processors[split],
            text_processor=self.text_processors[split],
            storage_path=storage,
        ).inner_dataset

        return dataset
