import torch
from typing import Union
from monai.data import ImageDataset
from AugmentedDataLoader.loaders import BaseAugmentedDataLoader

class AugmentedDataLoader(BaseAugmentedDataLoader):
    def __init__(
        self,
        dataset: ImageDataset,
        augmentation_transforms: list,
        batch_size: int = 1,
        subset_len: int = 1,
        transformation_device: Union[str, torch.device] = "cpu",
        return_on_device: Union[str, torch.device] = "cpu",
        debug_path: str = None,
        shuffle_mode: str = "full"
    ):
        self._dataset = dataset
        super().__init__(augmentation_transforms, batch_size, subset_len, transformation_device, return_on_device, debug_path, shuffle_mode)

    @property
    def dataset(self):
        return self._dataset

    @property
    def num_imgs(self):
        return len(self._dataset.image_files)

    @property
    def has_segmentations(self):
        return bool(self._dataset.seg_files)