import torch
import os
import random
from monai.data import ImageDataset
from utils.AugmentedDataLoaderUtils import save_subplot

"""
    -dataset -> ImageDataset type dataset, contains: images, labels and systematic transformations
    -augmentation_transforms -> List of (M) MONAI transformations for augmentation
    -batch_size -> Batch size (K) i.e. the batches to return
    -num_imgs -> Total number of images (N)
    -subset_len -> Subset length (J)
    -[optional] transformation_device -> if indicated, it is the device on which the user chooses to direct the transformation
    -[optional] return_device -> if indicated, it is the device to which the user chooses to direct each returned batch
    -[optional] debug_path -> if indicated, it is the path where the user chooses to save a slice of the image, for each image of the returned batches
"""

class AugmentedDataLoader:
    """
    Returns for each batch (example with batch_size = 2):
        batch = [
            0: torch.tensor (augmented image), torch.tensor (augmented mask or original label)
            1: torch.tensor (augmented image), torch.tensor (augmented mask or original label)
        ]
    """
    def __init__(
        self,
        dataset: ImageDataset,
        augmentation_transforms: list,
        batch_size: int,
        subset_len: int,
        transformation_device: str = "cpu",
        return_device: str = "cpu",
        debug_path: str = None,
    ):
        self.dataset = dataset  # ImageDataset type dataset, contains: images, segmentations or labels, systematic transformations
        self.augmentation_transforms = augmentation_transforms  # List of (M) MONAI transformations for augmentation
        self.batch_size = batch_size  # Batch size (K)
        self.num_imgs = len(dataset.image_files)  # Total number of images (N)
        self.subset_len = subset_len  # Subset length (J)
        self.debug_path = debug_path  # if indicated, it is the path where the user chooses to save a slice of the image, for each image of the returned batches
        self.transformation_device = transformation_device # if indicated, it is the device on which the user chooses to direct the transformation
        self.return_device = return_device # if indicated, this is the device to which the user chooses to direct each returned batch
        self.has_segmentations = bool(dataset.seg_files) # so as not to transform the labels
        
        if self.dataset is None:
            raise Exception("Dataset is None")
        
        if self.batch_size is None or self.batch_size == 0:
            raise Exception("Invalid batch size")
        
        if self.subset_len is None or self.batch_size == 0:
            raise Exception("Invalid subset len")
    
        
        
    def __iter__(self):
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
                    augmented_image = transformation(data[0].to(self.transformation_device))
                    
                    # Only transform if it's not a label
                    augmented_segmentation = transformation(data[1].to(self.transformation_device)) if self.has_segmentations else data[1]
                    
                    augmented_data.append((augmented_image, augmented_segmentation))

                augmented_subset.append((data[0], data[1]))  # Add the NOT augmented images
                """
                At this point, not all the images have been moved to the device that is used to make the transformations,
                this is because all the images will be moved to the device that the user has chosen to assign at the end of the process (return_on_device)
                """
                augmented_subset.extend(augmented_data)  # Add the augmented images

            """
            Further shuffle of the (J*M)+J images, in this way it will receive mixed augmented and unaugmented images
            """
            full_batch_indices = torch.randperm(len(augmented_subset))
            augmented_subset = [(augmented_subset[idx][0], augmented_subset[idx][1]) for idx in full_batch_indices]

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
                # Return batches of size K.
                for data in batch:
                    img, seg_or_label = data
                    if(batch[0][0].dtype !=  torch.float32): # Ensure that the batch elements are floats by checking first
                        img = img.float().to(self.return_on_device)
                        seg_or_label = seg_or_label.float().to(self.return_on_device) if self.has_segmentations else seg_or_label.to(self.return_on_device)
                    else:
                        img = img.to(self.return_on_device)
                        seg_or_label = seg_or_label.to(self.return_on_device)
                    
                    if self.debug_path:
                        debug_image_path = os.path.join(self.debug_path, f"augmented_image_{image_count}.png")
                        save_subplot(image=img, path=debug_image_path, image_count=image_count)
                        image_count+=1
                        
                # At this point, the intermediate batch chosen by the user is returned   
                yield batch

            """
            After having passed all the (J*M)+J batch images of K, I increase the index and start again to read other J images
            """
            index += checked_subset_len
            
