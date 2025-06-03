import torch
import random
from typing import Union
from AugmentedDataLoader.datasets import ImageToImageDataset
from AugmentedDataLoader.utils.AugmentedDataLoaderUtils import save_subplot_i2i
from AugmentedDataLoader.loaders import BaseAugmentedDataLoader

class AugmentedImageToImageDataLoader(BaseAugmentedDataLoader):
    def __init__(self, 
                 dataset: ImageToImageDataset, 
                 augmentation_transforms: list, 
                 batch_size: int, 
                 subset_len: int, 
                 transformation_device: Union[str, torch.device] = "cpu", 
                 return_on_device: Union[str, torch.device] = "cpu", 
                 debug_path: str = None,
                 shuffle_mode: str = "full"):
        self._dataset = dataset
        super().__init__(augmentation_transforms, batch_size, subset_len, transformation_device, return_on_device, debug_path, shuffle_mode)

    @property
    def dataset(self):
        return self._dataset

    @property
    def num_imgs(self):
        return len(self._dataset.first_type_image_files)

    @property
    def has_segmentations(self):
        return bool(self._dataset.seg_files)

    def _pair_shuffle(self, list1, list2, list3):
        paired_list = list(zip(list1, list2, list3))
        random.shuffle(paired_list)
        shuffled_list1, shuffled_list2, shuffled_list3 = zip(*paired_list)
        return list(shuffled_list1), list(shuffled_list2), list(shuffled_list3)

    def _apply_transformation(self, data, transformation):
            if self.has_segmentations:
                stacked_augmented_images = transformation(
                    torch.cat([
                        data[0].to(self.transformation_device), 
                        data[1].to(self.transformation_device), 
                        data[2].to(self.transformation_device)
                    ], dim=0)
                ) 
                augmented_images = torch.chunk(stacked_augmented_images, 3, dim=0)
            else:
                stacked_augmented_images = transformation(
                    torch.cat([
                        data[0].to(self.transformation_device), 
                        data[1].to(self.transformation_device)
                    ], dim=0)
                )
                augmented_images = torch.chunk(stacked_augmented_images, 2, dim=0) 
            
            augmented_images = [img.to(self.return_on_device) for img in augmented_images]
            return augmented_images
    
    def _generate_batches(self, image_batch1, image_batch2, segmentation_batch):
        paired_image_batch1, paired_image_batch2, paired_segmentation_batch = self._pair_shuffle(image_batch1, image_batch2, segmentation_batch)
        batch_num = 0
        
        for i in range(0, len(paired_image_batch1), self.batch_size):
            batch_images1 = paired_image_batch1[i:i + self.batch_size]
            batch_images2 = paired_image_batch2[i:i + self.batch_size]
            batch_segmentations = paired_segmentation_batch[i:i + self.batch_size]
            
            batch_images1 = torch.stack(batch_images1, dim=0)
            batch_images2 = torch.stack(batch_images2, dim=0)
            batch_segmentations = torch.stack(batch_segmentations, dim=0)
            
            seg_or_label_batch = batch_segmentations if self.has_segmentations else torch.tensor(batch_segmentations).to(self.return_on_device)
            
            if self.debug_path: 
                save_subplot_i2i(batch_images1, batch_images2, self.debug_path, batch_num)
            batch_num += 1
            
            yield batch_images1, batch_images2, seg_or_label_batch
            del batch_images1, batch_images2, seg_or_label_batch
    
    def _iter_full(self):
        shuffle_imgs_indices = list(range(self.num_imgs))
        random.shuffle(shuffle_imgs_indices)

        index = 0
        while index < self.num_imgs:
            remaining_imgs = self.num_imgs - index
            checked_subset_len = min(self.subset_len, remaining_imgs)
            subset_indices = shuffle_imgs_indices[index : index + checked_subset_len]
            subset = torch.utils.data.Subset(self.dataset, subset_indices)
            
            augmented_image1_super_batch, augmented_image2_super_batch, augmented_segmentation_super_batch = [], [], []
            for data in subset:
                for transformation in self.augmentation_transforms:
                    augmented_images = self._apply_transformation(data, transformation)
                    augmented_image1_super_batch.append(augmented_images[0])
                    augmented_image2_super_batch.append(augmented_images[1])
                    if self.has_segmentations:
                        augmented_segmentation_super_batch.append(augmented_images[2])
                    else:
                        augmented_segmentation_super_batch.append(torch.tensor(data[2]).float().to(self.return_on_device))
                    
                    augmented_image1_super_batch.append(data[0].float().to(self.return_on_device))
                    augmented_image2_super_batch.append(data[1].float().to(self.return_on_device))
                    augmented_segmentation_super_batch.append(data[2].float().to(self.return_on_device) if self.has_segmentations else torch.tensor(data[2]).float().to(self.return_on_device))
            
            yield from self._generate_batches(augmented_image1_super_batch, augmented_image2_super_batch, augmented_segmentation_super_batch)
            del augmented_image1_super_batch, augmented_image2_super_batch, augmented_segmentation_super_batch
            index += checked_subset_len

    def _iter_pseudo(self):
        shuffle_imgs_indices = list(range(self.num_imgs))
        random.shuffle(shuffle_imgs_indices)

        index = 0
        while index < self.num_imgs:
            remaining_imgs = self.num_imgs - index
            checked_subset_len = min(self.batch_size, remaining_imgs)
            subset_indices = shuffle_imgs_indices[index: index + checked_subset_len]
            subset = torch.utils.data.Subset(self.dataset, subset_indices)

            for transformation in self.augmentation_transforms:                
                augmented_image1_batch, augmented_image2_batch, augmented_segmentation_batch = [], [], []
                
                for data in subset:
                    augmented_images = self._apply_transformation(data, transformation)
                    augmented_image1_batch.append(augmented_images[0])
                    augmented_image2_batch.append(augmented_images[1])
                    if self.has_segmentations:
                        augmented_segmentation_batch.append(augmented_images[2])
                    else:
                        augmented_segmentation_batch.append(torch.tensor(data[2]).float().to(self.return_on_device))
                
                yield from self._generate_batches(augmented_image1_batch, augmented_image2_batch, augmented_segmentation_batch)
                del augmented_image1_batch, augmented_image2_batch, augmented_segmentation_batch
            
            original_batch_img1 = [data[0].float().to(self.return_on_device) for data in subset]
            original_batch_img2 = [data[1].float().to(self.return_on_device) for data in subset]
            original_batch_seg = ([data[2].float().to(self.return_on_device) for data in subset]
                                  if self.has_segmentations 
                                  else [torch.tensor(data[2]).float().to(self.return_on_device) for data in subset])
            
            yield from self._generate_batches(original_batch_img1, original_batch_img2, original_batch_seg)
            del original_batch_img1, original_batch_img2, original_batch_seg
            index += checked_subset_len

    def __iter__(self):
        if self.shuffle_mode == "full":
            return self._iter_full()
        if self.shuffle_mode == "pseudo":
            return self._iter_pseudo()