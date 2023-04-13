from monai.data import (
    CacheDataset,
    DataLoader
)
from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    LoadImaged,
    Zoomd,
    RandRotated,
)
import matplotlib.pyplot as plt, SimpleITK as sitk


def get_transforms(transform):
    return Compose(
                [
                    LoadImaged(keys=["image", "label"], image_only=True),
                    EnsureChannelFirstd(keys=["image", "label"]),
                    transform,
                ]
            )

def get_dict_list(images, labels):
    return [
    {"image": image, "label": label} for image, label in zip(images, labels)
]      

def augment(images,labels,transforms):
    data_dict_list = get_dict_list(images, labels)
    print(f"paths: {data_dict_list}")

    augmented_images = []
    for _, transform in enumerate(transforms):
        temp_transform = get_transforms(transform)
        cache_augm_ds = CacheDataset(
        data=data_dict_list,
        transform=temp_transform,
        cache_rate=1.0,
        runtime_cache="processes",
        copy_cache=False,
        ),

        for img in cache_augm_ds:
            data = DataLoader(img, batch_size=1, num_workers=4)
            [augmented_images.append(batch_data) for batch_data in data] 
    return augmented_images


# example data
images = ["./data/MedNIST/AbdomenCT/000000.jpeg", "./data/MedNIST/AbdomenCT/000000.jpeg", "./data/MedNIST/AbdomenCT/000000.jpeg", "./data/MedNIST/AbdomenCT/000000.jpeg", "./data/MedNIST/AbdomenCT/000000.jpeg", "./data/MedNIST/AbdomenCT/000000.jpeg", "./data/MedNIST/AbdomenCT/000000.jpeg", "./data/MedNIST/AbdomenCT/000000.jpeg", "./data/MedNIST/AbdomenCT/000000.jpeg", "./data/MedNIST/AbdomenCT/000000.jpeg", "./data/MedNIST/AbdomenCT/000000.jpeg", "./data/MedNIST/AbdomenCT/000000.jpeg", "./data/MedNIST/AbdomenCT/000000.jpeg", "./data/MedNIST/AbdomenCT/000000.jpeg", "./data/MedNIST/AbdomenCT/000000.jpeg", "./data/MedNIST/AbdomenCT/000000.jpeg", "./data/MedNIST/AbdomenCT/000000.jpeg", "./data/MedNIST/AbdomenCT/000000.jpeg", "./data/MedNIST/AbdomenCT/000000.jpeg", "./data/MedNIST/AbdomenCT/000000.jpeg", "./data/MedNIST/AbdomenCT/000000.jpeg", "./data/MedNIST/AbdomenCT/000000.jpeg", "./data/MedNIST/AbdomenCT/000000.jpeg", "./data/MedNIST/AbdomenCT/000000.jpeg", "./data/MedNIST/AbdomenCT/000000.jpeg", "./data/MedNIST/AbdomenCT/000000.jpeg", "./data/MedNIST/AbdomenCT/000000.jpeg", "./data/MedNIST/AbdomenCT/000000.jpeg"]
labels = ["./data/MedNIST/AbdomenCT/000000.jpeg", "./data/MedNIST/AbdomenCT/000000.jpeg", "./data/MedNIST/AbdomenCT/000000.jpeg", "./data/MedNIST/AbdomenCT/000000.jpeg", "./data/MedNIST/AbdomenCT/000000.jpeg", "./data/MedNIST/AbdomenCT/000000.jpeg", "./data/MedNIST/AbdomenCT/000000.jpeg", "./data/MedNIST/AbdomenCT/000000.jpeg", "./data/MedNIST/AbdomenCT/000000.jpeg", "./data/MedNIST/AbdomenCT/000000.jpeg", "./data/MedNIST/AbdomenCT/000000.jpeg", "./data/MedNIST/AbdomenCT/000000.jpeg", "./data/MedNIST/AbdomenCT/000000.jpeg", "./data/MedNIST/AbdomenCT/000000.jpeg", "./data/MedNIST/AbdomenCT/000000.jpeg", "./data/MedNIST/AbdomenCT/000000.jpeg", "./data/MedNIST/AbdomenCT/000000.jpeg", "./data/MedNIST/AbdomenCT/000000.jpeg", "./data/MedNIST/AbdomenCT/000000.jpeg", "./data/MedNIST/AbdomenCT/000000.jpeg", "./data/MedNIST/AbdomenCT/000000.jpeg", "./data/MedNIST/AbdomenCT/000000.jpeg", "./data/MedNIST/AbdomenCT/000000.jpeg", "./data/MedNIST/AbdomenCT/000000.jpeg", "./data/MedNIST/AbdomenCT/000000.jpeg", "./data/MedNIST/AbdomenCT/000000.jpeg", "./data/MedNIST/AbdomenCT/000000.jpeg", "./data/MedNIST/AbdomenCT/000000.jpeg"]
transforms = [
    Zoomd(keys=["image", "label"], zoom=0.4),
    RandRotated(keys=["image", "label"], prob=1, range_x=[0.4, 0.4]),
]

print(augment(images,labels,transforms))