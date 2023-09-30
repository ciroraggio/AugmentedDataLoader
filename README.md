# AugmentedDataLoader
Medical image augmentation tool that can be integrated with Pytorch & MONAI, by Ciro B. Raggio and P. Zaffino.

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

augmented_data_loader = AugmentedDataLoader(dataset, 
                                            augmentation_transforms, 
                                            batch_size, 
                                            subset_size, 
                                            transformation_device, # cuda device 0
                                            return_device, # cuda device 1
                                            debug_path) # use debug 

for batch_data in augmented_data_loader:
        augmented_imgs, augmented_segms = batch_data[0], batch_data[1]
        # network training
        # ...

```

## Workflow
![AugmentedDataLoaderWorkflow](/assets/workflow.png)

