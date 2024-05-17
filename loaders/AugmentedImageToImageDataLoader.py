import torch
import os
import random
from typing import List, Union
from datasets.ImageToImageDataset import ImageToImageDataset
from utils.AugmentedDataLoaderUtils import save_subplot

class AugmentedImageToImageDataLoader:
    """
    This class was created to apply AugmentedDataLoader logic to an ImageToImageDataset composed of an image/mask triplet.
    Manages 3 images at a time, for instance, use cases can be: CT.nrrd, MRI.nrrd, mask.nrrd    
    Returns for each batch (example with batch_size = 2):
        batch = [
            0: torch.tensor (augmented first image type), torch.tensor (augmented second image type), torch.tensor (augmented mask or original label)
            1: torch.tensor (augmented first image type), torch.tensor (augmented second image type), torch.tensor (augmented mask or original label)
        ]
    """
    def __init__(self, 
                 dataset: ImageToImageDataset, 
                 augmentation_transforms: list, 
                 batch_size: int, 
                 subset_len: int, 
                 transformation_device: str = "cpu", 
                 return_on_device: str = "cpu", 
                 debug_path: str = None):

        self.dataset = dataset  # ImageDataset type dataset, contains: images, segmentations or labels, systematic transformations
        self.augmentation_transforms = augmentation_transforms  # List of (M) MONAI transformations for augmentation
        self.batch_size = batch_size  # Batch size (K)
        self.num_imgs = len(dataset.first_type_image_files)  # Total number of images (N)
        self.subset_len = subset_len  # Subset length (J)
        self.debug_path = debug_path  # if indicated, it is the path where the user chooses to save a slice of the image, for each image of the returned batches
        self.transformation_device = transformation_device # if indicated, it is the device on which the user chooses to direct the transformation
        self.return_on_device = return_on_device # if indicated, this is the device to which the user chooses to direct each returned batch
        self.has_segmentations = bool(dataset.seg_files) # so as not to transform the labels
    
    def __len__(self) -> int:
        return len(self.dataset.first_type_image_files)
    
    def __iter__(self) -> List[tuple[torch.Tensor, torch.Tensor, Union[torch.Tensor, None]]]:
        # Create a list containing all the indexes of the images and apply the shuffle so as not to operate on the images in the same order of arrival
        shuffle_imgs_indices = list(range(self.num_imgs))
        random.shuffle(shuffle_imgs_indices)

        index = 0
        while index < self.num_imgs:
            """
            checked_subset_len determines the length of the subset that will be extracted,
            if the user-specified value for subset_len is greater than the number of images remaining,
            then subset_len will be equal to the number of remaining images, so as to avoid exceeding the length of the image list
            """
            remaining_imgs = self.num_imgs - index
            checked_subset_len = min(self.subset_len, remaining_imgs)

            """
            A subset of images is created (which are chosen from the list of previously mixed indices) having length J            
            """
            subset_indices = shuffle_imgs_indices[index : index + checked_subset_len]
            subset = torch.utils.data.Subset(self.dataset, subset_indices)
            
            augmented_subset = []
            for data in subset:
                augmented_data = []
                for transformation in self.augmentation_transforms:
                    # Only stack and transform together if data[2] it's not a label
                    if self.has_segmentations:
                        # the torch.cat is used to prevent mismatching between the images in case of random transformations
                        stacked_augmented_images = transformation(
                                                        torch.cat([
                                                            data[0].to(self.transformation_device), 
                                                            data[1].to(self.transformation_device), 
                                                            data[2].to(self.transformation_device)
                                                        ], dim=0)
                                                    ) 
                        augmented_first_image, augmented_second_image, augmented_segmentation = torch.chunk(stacked_augmented_images, 3, dim=0)
                    else:
                        stacked_augmented_images = transformation(torch.cat(data[0].to(self.transformation_device), data[1].to(self.transformation_device)))
                        augmented_first_image, augmented_second_image = torch.chunk(stacked_augmented_images, 2, dim=0) 
                        augmented_segmentation = data[2]

                    augmented_data.append((augmented_first_image, augmented_second_image, augmented_segmentation))

                augmented_subset.append((data[0], data[1], data[2]))  # Add the NOT augmented images
                """
                At this point, not all the images have been moved to the device that is used to make the transformations,
                this is because all the images will be moved to the device that the user has chosen to assign at the end of the process (return_on_device)
                """
                augmented_subset.extend(augmented_data)  # Add the augmented images

            """
            Further shuffle of the (J*M)+J images, in this way it will receive mixed augmented and unaugmented images
            """
            full_batch_indices = torch.randperm(len(augmented_subset))
            augmented_subset = [(augmented_subset[idx][0], augmented_subset[idx][1], augmented_subset[idx][2]) for idx in full_batch_indices]

            """
            Create batches of size K
            -range(0, len(augmented_subset), self.batch_size):
                generates a sequence of values ​​representing the start indices of each batch.
                The indices start from 0 and advance with a step equal to self.batch_size, until reaching the total length of augmented_subset.
            -augmented_subset[i:i + self.batch_size]:
                selects a sub-list of augmented_subset ranging from index i to index i + self.batch_size.
                This creates a batch of images of size self.batch_size.
            """
            batches = [
                augmented_subset[i : i + self.batch_size]
                for i in range(0, len(augmented_subset), self.batch_size)
            ]
            
            image_count = 0
            for batch in batches:
                for data in batch:
                    first_type_img, second_type_img, seg_or_label = data
                    if(batch[0][0].dtype !=  torch.float32): # Ensure that the batch elements are floats by checking first
                        first_type_img = first_type_img.float().to(self.return_on_device)
                        second_type_img = second_type_img.float().to(self.return_on_device)
                        seg_or_label = seg_or_label.float().to(self.return_on_device) if self.has_segmentations else seg_or_label.to(self.return_on_device)
                    else:
                        first_type_img = first_type_img.to(self.return_on_device)
                        second_type_img = second_type_img.to(self.return_on_device)
                        seg_or_label = seg_or_label.to(self.return_on_device)
                    
                    if self.debug_path:
                        debug_image_path = os.path.join(self.debug_path, f"augmented_image_{image_count}.png")
                        save_subplot(image=first_type_img, path=debug_image_path, image_count=image_count)
                        image_count+=1
                        
                # At this point, the intermediate batch chosen by the user is returned   
                yield batch
                
            # After having passed all the (J*M)+J batch images of K, increase the index and start again to read other J images    
            index += checked_subset_len