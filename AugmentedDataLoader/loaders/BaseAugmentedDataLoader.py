import os
import torch
import random
from abc import ABC, abstractmethod
from typing import Union, List
from AugmentedDataLoader.utils.AugmentedDataLoaderUtils import save_subplot

SHUFFLE_MODE_LIST = ["full", "pseudo"]

class BaseAugmentedDataLoader(ABC):
    def __init__(
        self,
        augmentation_transforms: list,
        batch_size: int,
        subset_len: int,
        transformation_device: Union[str, torch.device],
        return_on_device: Union[str, torch.device],
        debug_path: str,
        shuffle_mode: str
    ):
        self.augmentation_transforms = augmentation_transforms
        self.batch_size = batch_size
        self.subset_len = subset_len
        self.debug_path = debug_path
        self.transformation_device = transformation_device
        self.return_on_device = return_on_device
        self.shuffle_mode = shuffle_mode

        if self.dataset is None:
            raise Exception("Dataset is None")
        
        if self.batch_size == 0:
            raise Exception("Invalid batch size")

        if self.shuffle_mode == "full" and (self.subset_len is None or self.subset_len == 0):
            raise Exception("Invalid subset len")

        if self.batch_size is None or self.batch_size == 0:
            raise Exception("Invalid batch size")
        
        if self.augmentation_transforms is None or len(self.augmentation_transforms) == 0:
            raise Exception("The augmentation_transforms list must not be empty")

        if self.shuffle_mode not in SHUFFLE_MODE_LIST:
            raise Exception(f"Mode '{self.shuffle_mode}' not supported")

        if self.has_segmentations and bool(self.dataset.labels):
            raise Exception(f"Found segmentations and labels at the same time")
        
        if self.debug_path and not os.path.exists(self.debug_path):
            os.makedirs(self.debug_path)

    @property
    @abstractmethod
    def dataset(self):
        pass

    @property
    @abstractmethod
    def num_imgs(self):
        pass

    @property
    @abstractmethod
    def has_segmentations(self):
        pass
    
    def __len__(self) -> int:
        return self.num_imgs   

    def _pair_shuffle(self, list1, list2):
        paired_list = list(zip(list1, list2))
        random.shuffle(paired_list)
        shuffled_list1, shuffled_list2 = zip(*paired_list)
        return list(shuffled_list1), list(shuffled_list2)
    
    def _apply_transformation(self, data, transformation):
        if self.has_segmentations:
            stacked_augmented_images = transformation(
                torch.cat([
                    data[0].float().to(self.transformation_device), 
                    data[1].float().to(self.transformation_device), 
                ], dim=0)
            ) 
            augmented_image, augmented_segmentation = torch.chunk(stacked_augmented_images, 2, dim=0)
        else:
            augmented_image = transformation(data[0].float().to(self.transformation_device))
            augmented_segmentation = torch.tensor(data[1]).float()
        
        return augmented_image.to(self.return_on_device), augmented_segmentation.to(self.return_on_device)
    
    def _generate_batches(self, image_batch, segmentation_batch):
        image_batch, segmentation_batch = self._pair_shuffle(image_batch, segmentation_batch)
        batch_num = 0
        
        for i in range(0, len(image_batch), self.batch_size):
            batch_images = image_batch[i:i + self.batch_size]
            batch_segmentations = segmentation_batch[i:i + self.batch_size]
            
            batch_images = torch.stack(batch_images, dim=0)
            batch_segmentations = torch.stack(batch_segmentations, dim=0)
            
            seg_or_label_batch = batch_segmentations if self.has_segmentations else torch.tensor(batch_segmentations)
            
            if self.debug_path: 
                save_subplot(batch_images, self.debug_path, batch_num)
            batch_num += 1
            
            yield batch_images, seg_or_label_batch
            del batch_images, batch_segmentations
    
    def _iter_full(self):
        shuffle_imgs_indices = list(range(self.num_imgs))
        random.shuffle(shuffle_imgs_indices)

        index = 0
        while index < self.num_imgs:
            remaining_imgs = self.num_imgs - index
            checked_subset_len = min(self.subset_len, remaining_imgs)
            subset_indices = shuffle_imgs_indices[index : index + checked_subset_len]
            subset = torch.utils.data.Subset(self.dataset, subset_indices)
            
            augmented_image_super_batch, augmented_segmentation_super_batch = [], []
            for data in subset:
                for transformation in self.augmentation_transforms:
                    augmented_image, augmented_segmentation = self._apply_transformation(data, transformation)
                    augmented_image_super_batch.append(augmented_image)
                    augmented_segmentation_super_batch.append(augmented_segmentation)
                
                augmented_image_super_batch.append(data[0].float().to(self.return_on_device))
                augmented_segmentation_super_batch.append(data[1].float().to(self.return_on_device) if self.has_segmentations else torch.tensor(data[1]).float().to(self.return_on_device))

            yield from self._generate_batches(augmented_image_super_batch, augmented_segmentation_super_batch)
            del augmented_image_super_batch, augmented_segmentation_super_batch
            index += checked_subset_len
    
    def _iter_pseudo(self):
        shuffle_imgs_indices = list(range(self.num_imgs))
        random.shuffle(shuffle_imgs_indices)

        index = 0
        while index < self.num_imgs:
            remaining_imgs = self.num_imgs - index
            checked_subset_len = min(self.batch_size, remaining_imgs)
            subset_indices = shuffle_imgs_indices[index : index + checked_subset_len]
            subset = torch.utils.data.Subset(self.dataset, subset_indices)

            for transformation in self.augmentation_transforms:                
                augmented_image_batch, augmented_segmentation_batch = [], []
                
                for data in subset:
                    augmented_image, augmented_segmentation = self._apply_transformation(data, transformation)
                    augmented_image_batch.append(augmented_image)
                    augmented_segmentation_batch.append(augmented_segmentation)
                
                yield from self._generate_batches(augmented_image_batch, augmented_segmentation_batch)
                del augmented_image_batch, augmented_segmentation_batch
            
            original_batch_img = [data[0].float().to(self.return_on_device) for data in subset]
            original_batch_seg = ([data[1].float().to(self.return_on_device) for data in subset]
                                  if self.has_segmentations 
                                  else [torch.tensor(data[1]).float().to(self.return_on_device) for data in subset])
            
            yield from self._generate_batches(original_batch_img, original_batch_seg)
            del original_batch_img, original_batch_seg
            index += checked_subset_len
        
    def __iter__(self):
        if self.shuffle_mode == "full":
            return self._iter_full()
        if self.shuffle_mode == "pseudo":
            return self._iter_pseudo()