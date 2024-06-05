import torch
import os
import random
from monai.data import ImageDataset
from AugmentedDataLoader.utils.AugmentedDataLoaderUtils import save_subplot

SHUFFLE_MODE_LIST=["full", "pseudo"]

"""
    -dataset -> ImageDataset type dataset, contains: images, labels and systematic transformations
    -augmentation_transforms -> List of (M) MONAI transformations for augmentation
    -batch_size -> Batch size (K) i.e. the batches to return
    -num_imgs -> Total number of images (N)
    -subset_len -> Subset length (J)
    -[optional] transformation_device -> if indicated, it is the device on which the user chooses to direct the transformation
    -[optional] return_on_device -> if indicated, it is the device to which the user chooses to direct each returned batch
    -[optional] debug_path -> if indicated, it is the path where the user chooses to save a slice of the image, for each image of the returned batches
    -[optional] shuffle_mode -> available options: 
                                - full: each transformation is applied to each image in the subset by keep it in memory. 
                                          The possible combinations are then mixed, including the original images, and the batches are returned.
                                          It requires more memory, but is faster and more generalizable.
                                    
                                - pseudo: the logic is based only on the batch size, which is shuffled and returned immediately after each transformation.
                                        It takes up less memory but is slower and less generalizable.

"""
class AugmentedDataLoader:
    def __init__(
        self,
        dataset: ImageDataset,
        augmentation_transforms: list,
        batch_size: int,
        subset_len: int,
        transformation_device: str = "cpu",
        return_on_device: str = "cpu",
        debug_path: str = None,
        shuffle_mode: str = "full"
    ):
        self.dataset = dataset  # ImageDataset type dataset, contains: images, segmentations or labels, systematic transformations
        self.augmentation_transforms = augmentation_transforms  # List of (M) MONAI transformations for augmentation
        self.batch_size = batch_size  # Batch size (K)
        self.num_imgs = len(dataset.image_files)  # Total number of images (N)
        self.subset_len = subset_len  # Subset length (J)
        self.debug_path = debug_path  # if indicated, it is the path where the user chooses to save a slice of the image, for each image of the returned batches
        self.transformation_device = transformation_device # if indicated, it is the device on which the user chooses to direct the transformation
        self.return_on_device = return_on_device # if indicated, this is the device to which the user chooses to direct each returned batch
        self.has_segmentations = bool(dataset.seg_files) # so as not to transform the labels
        """shuffle_mode - full: each transformation is applied to each image in the subset by keep it in memory. 
                                                     The possible combinations are then mixed, including the original images, and the batches are returned.
                                                     It requires more memory, but is faster and more generalizable.
                                    
                                            - pseudo: subset_len parameter will have no effect. 
                                                      The logic is based only on the batch size and the  which is shuffled and returned immediately after each transformation.
                                                      It takes up less memory but is slower and less generalizable.
        """
        self.shuffle_mode = shuffle_mode 
        
        if self.dataset is None:
            raise Exception("Dataset is None")
        
        if self.batch_size is None or self.batch_size == 0:
            raise Exception("Invalid batch size")
        
        if self.shuffle_mode == "full" and (self.subset_len is None or self.subset_len == 0):
            raise Exception("Invalid subset len")
        
        if self.shuffle_mode not in SHUFFLE_MODE_LIST:
            raise Exception(f"Mode '{self.shuffle_mode}' not supported")

        if self.has_segmentations and bool(dataset.labels):
            raise Exception(f"Found segmentations and labels at the same time")
    
    def __len__(self) -> int:
        return self.num_imgs   

    def _pair_shuffle(self, list1, list2):
        paired_list = list(zip(list1, list2))
        random.shuffle(paired_list)
        shuffled_list1, shuffled_list2 = zip(*paired_list)
        return shuffled_list1, shuffled_list2
    
    def _iter_full(self):
        """
        NOTE: 
        In this shuffle mode, each transformation is applied to each image in the subset by keep it in memory. 
        The possible combinations are then mixed, including the original images, and the batches are returned.
        It requires more memory, but is faster and more generalizable.
        """
        # Create a list containing all the indexes of the images and apply the shuffle so as not to operate on the images in the original order 
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
            
            augmented_image_super_batch, augmented_segmentation_super_batch = [], []
            for data in subset:
                for transformation in self.augmentation_transforms:
                    # Only stack and transform together if data[1] it's not a label
                    if self.has_segmentations:
                        # the torch.cat is used to prevent mismatching between the images in case of random transformations
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

                    # Add augmented images and labels
                    augmented_image_super_batch.append(augmented_image.to(self.return_on_device))
                    augmented_segmentation_super_batch.append(augmented_segmentation.to(self.return_on_device))

                    # Add original images and labels (unchanged)
                    augmented_image_super_batch.append(data[0].float().to(self.return_on_device))
                    augmented_segmentation_super_batch.append(data[1].float().to(self.return_on_device) if self.has_segmentations else torch.tensor(data[1]).float())

            """
            Further shuffle of the (J*M)+J images, in this way the model will receive mixed augmented and non-augmented images
            """
            augmented_image_super_batch, augmented_segmentation_super_batch = self._pair_shuffle(augmented_image_super_batch, augmented_segmentation_super_batch)
            # Create batches
            for i in range(0, len(augmented_image_super_batch), self.batch_size):
                batch_images = augmented_image_super_batch[i:i + self.batch_size]
                batch_segmentations = augmented_segmentation_super_batch[i:i + self.batch_size]
                
                # TODO Add save_subplot
                seg_or_label_batch = torch.cat(batch_segmentations, dim=0) if self.has_segmentations else torch.tensor(batch_segmentations).to(self.return_on_device)

                yield torch.cat(batch_images, dim=0), seg_or_label_batch
                del batch_images, batch_segmentations # Delete the yielded batches to free up the memory

            """
            After having passed all the (J*M)+J batch images of K, I increase the index and start again to read other J images (a new subset)
            """
            del augmented_image_super_batch, augmented_segmentation_super_batch 
            index += checked_subset_len
    
    def _iter_pseudo(self):
        """
        NOTE: 
        In this shuffle mode, the logic is based only on the batch size, which is shuffled and returned immediately after each transformation.
        It takes up less memory but is slower and less generalizable.
        """

        shuffle_imgs_indices = list(range(self.num_imgs))
        random.shuffle(shuffle_imgs_indices)

        index = 0
        while index < self.num_imgs:
            # Subset extraction
            remaining_imgs = self.num_imgs - index
            checked_subset_len = min(self.batch_size, remaining_imgs)
            subset_indices = shuffle_imgs_indices[index : index + checked_subset_len]
            subset = torch.utils.data.Subset(self.dataset, subset_indices)

            original_image_batch, original_segmentation_batch = [], []
            for transformation in self.augmentation_transforms:                
                augmented_image_batch, augmented_segmentation_batch = [], []
                
                for data in subset:
                    # Only stack and transform together if data[1] it's not a label
                    if self.has_segmentations:
                        # the torch.cat is used to prevent mismatching between the images in case of random transformations
                        stacked_augmented_images = transformation(
                                                        torch.cat([
                                                            data[0].float().to(self.transformation_device), 
                                                            data[1].float().to(self.transformation_device), 
                                                        ], dim=0)
                                                    ) 
                        augmented_image, augmented_segmentation = torch.chunk(stacked_augmented_images, 2, dim=0)
                        augmented_image, augmented_segmentation = augmented_image.to(self.return_on_device), augmented_segmentation.to(self.return_on_device) 
                    else:
                        augmented_image, augmented_segmentation = transformation(data[0].float().to(self.transformation_device)).to(self.return_on_device), torch.tensor(data[1]).float()

                    augmented_image_batch.append(augmented_image)
                    augmented_segmentation_batch.append(augmented_segmentation)

                # Combined shuffle to avoid mismatch between images and masks
                augmented_image_batch, augmented_segmentation_batch = self._pair_shuffle(augmented_image_batch, augmented_segmentation_batch)
                
                # TODO Add save_subplot
                seg_or_label_batch = torch.cat(augmented_segmentation_batch, dim=0) if self.has_segmentations else torch.tensor(augmented_segmentation_batch).to(self.return_on_device)

                yield torch.cat(augmented_image_batch, dim=0), seg_or_label_batch
                del augmented_image_batch, augmented_segmentation_batch, seg_or_label_batch  # Delete the yielded batches to free up the memory

            # yield the original data with the same "collect, mix, yield" logic before changing subset
            original_batch_img = [data[0].float().to(self.return_on_device) for data in subset]
            original_batch_seg = ([data[1].float().to(self.return_on_device) for data in subset]
                                    if self.has_segmentations 
                                    else [torch.tensor(data[1]).float() for data in subset])
            
            original_batch_img, original_batch_seg = self._pair_shuffle(original_batch_img, original_batch_seg)
            original_seg_or_label_batch = torch.cat(original_batch_seg, dim=0) if self.has_segmentations else torch.tensor(original_batch_seg).to(self.return_on_device)

            yield torch.cat(original_batch_img, dim=0), original_seg_or_label_batch
            del original_batch_img, original_batch_seg, original_seg_or_label_batch

            index += checked_subset_len
        
    def __iter__(self):
        if self.shuffle_mode == "full":
            return self._iter_full()
        if self.shuffle_mode == "pseudo":
            return self._iter_pseudo()
