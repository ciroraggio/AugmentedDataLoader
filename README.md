# AugmentedDataLoader
Medical image augmentation tool that can be integrated with Pytorch & MONAI, by Ciro B. Raggio and P. Zaffino.

- [AugmentedDataLoader](#augmenteddataloader)
  - [Description](#description)
  - [How it works](#how-it-works)
  - [How to use it](#how-to-use-it)
    - [AugmentedDataLoader](#augmenteddataloader-1)
    - [AugmentedImageToImageDataLoader](#augmentedimagetoimagedataloader)
- [Workflow](#workflow)
- [Changelog](#changelog)
  
## Description
A medical image augmentation algorithm developed to improve flexibility and customization in the data augmentation process for medical image processing applications. It can be integrated and used within the MONAI framework.

It's designed to operate on a dataset of medical images and apply a series of specific transformations to each image. This process augments the original dataset **on-the-fly**, providing a greater variety of samples for training deep learning models.

We recommend the use of ImageDataset available in MONAI, which automatically handles the loading of images and associated segmentations/labels.

## How it works

**Configuration Parameters:**
- Define a list of MONAI transformations to be applied to each image. These transformations can include rotation, scaling, cropping, and other operations.
- Specify the size of the subset of images to be transformed at each iteration.
- Set the batch size to determine how many samples will be returned in each block.

**Optional Parameters:**
- You can specify the device on which to perform the transformations and on which device to address the returned blocks.
- If necessary, define a debug path to save a slice of the image of each augmented sample for debugging purposes.
  
<br>

## How to use it
Here are some simple snippets ready to use.

### AugmentedDataLoader
You just declare:
- an ImageDataset 
- how many images to process and how many to receive at each iteration 
- MONAI transformations to apply to the images for the augmentation process. 

Declare an AugmentedDataLoader and receive the augmented images on the fly!

```python
from monai.transforms import Rotate, Flip, Compose, Resize
from monai.data import ImageDataset
from AugmentedDataLoader import AugmentedDataLoader

# ImageDataset params
images_to_transform = [...]
seg_to_transform = [...]
each_image_trans = Compose([
                            Resize([74,74,1])
                            ])

# AugmentedDataLoader params
augmentation_transforms = [
    Rotate(angle=[0.4, 0.4,0.4]), # 0.4 rad
    Flip(spatial_axis=1),
]
subset_size = 2
batch_size = 2
debug_path='./debug_path_example'
transformation_device=0
return_device=1

dataset = ImageDataset(image_files=images_to_transform, seg_files=seg_to_transform, transform=each_image_trans, seg_transform=each_image_trans)

augmented_data_loader = AugmentedDataLoader(dataset=dataset, # Plain ImageDataset
                                            augmentation_transforms=augmentation_transforms, 
                                            batch_size=batch_size, 
                                            subset_len=subset_size, 
                                            transformation_device=transformation_device, 
                                            debug_path) # use debug 

for batches in augmented_data_loader:
    for batch in batches:
        augm_img, augm_mask_or_label = batch
        # network training
        # ...

```

### AugmentedImageToImageDataLoader
In this case, declare:
- an **ImageToImageDataset**, imported from ***datasets***, which accepts three images (image, image, mask or label) and transformations to apply to the three types
- how many images to process and how many to receive at each iteration
- MONAI transformations to apply to the images for the augmentation process. 

Declare an AugmentedDataLoader and receive the augmented images on the fly!

```python
import os
from monai.transforms import Flip, Compose, Resize
from loaders.AugmentedImageToImageDataLoader import AugmentedImageToImageDataLoader
from datasets.ImageToImageDataset import ImageToImageDataset

first_type_image_path = "..."
second_type_image_path = "..."
seg_path = "..."

# ImageDataset params
first_type_images_to_transform = [f"{first_type_image_path}/{img}" for img in os.listdir(first_type_image_path)]
second_type_images_to_transform = [f"{second_type_image_path}/{img}" for img in os.listdir(second_type_image_path)]
seg_to_transform = [f"{seg_path}/{mask}" for mask in os.listdir(seg_path)]

each_image_trans = Compose([
    # ...transformations to apply to all images here...
])

# AugmentedDataLoader params
augmentation_transforms = [
    Flip(spatial_axis=1),
]

subset_size = 2
batch_size = 2
transformation_device="cuda:2"
return_device="cuda:2"

dataset = ImageToImageDataset(first_type_image_files=first_type_images_to_transform,
                              second_type_image_files=second_type_images_to_transform,
                              seg_files=seg_to_transform, 
                              first_type_image_transforms=each_image_trans, 
                              second_type_image_transforms=each_image_trans, # can be different
                              seg_transform=each_image_trans, # can be different
                              reader="pilreader"  # can be different
                              )

augmented_data_loader = AugmentedImageToImageDataLoader(
                                            dataset=dataset, # Custom ImageToImageDataset
                                            augmentation_transforms=augmentation_transforms, 
                                            batch_size=batch_size, 
                                            subset_len=subset_size, 
                                            transformation_device=transformation_device, 
                                            return_on_device=return_device
                                            )
for batches in augmented_data_loader:
    for batch in batches:
        augm_first_type_image, augm_second_type_image, augm_mask_or_label = batch
        # network training
        # ...
    
```
# Workflow![AugmentedDataLoaderWorkflow](/assets/workflow.png)

# Changelog
- v2.0:
    - Added a new dataset (ImageToImageDataset) and a new dataloader (AugmentedImageToImageDataLoader) to support cases in which the user wants to augment two images associated with a simultaneous mask/segmentation
    - Fixed bugs on returning batches in AugmentedDataLoader
    - Updated documentation
- v1.0 Refactorings and bug fixes
- v1.0a AugmentedDataLoader released