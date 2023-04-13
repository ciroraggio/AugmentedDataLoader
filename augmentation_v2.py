from monai.data import (
    CacheDataset,
    DataLoader,
)
from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    LoadImaged,
    Orientationd,
)
import matplotlib.pyplot as plt
import SimpleITK as sitk

def get_transforms(transforms):
    return Compose(
        [
            LoadImaged(keys=["image", "label"], image_only=True),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            transforms
        ]
    )
    
data_list = [
    {"image": "./data/MedNIST/AbdomenCT/000000.jpeg", "label": "./data/MedNIST/AbdomenCT/000001.jpeg"},
    {"image": "./data/MedNIST/AbdomenCT/000000.jpeg", "label": "./data/MedNIST/AbdomenCT/000001.jpeg"},
]

transforms = []

trans = get_transforms(transforms)

cache_augm_ds = CacheDataset(
    data=data_list, transform=trans, cache_rate=1.0, runtime_cache="processes", copy_cache=False
)

data = DataLoader(
        cache_augm_ds,
        batch_size=2,
        shuffle=True,
        num_workers=0,
    )

# for batch_data in data:
#   # Plot the images using matplotlib
#   fig, ax = plt.subplots(1, 2)
#   ax[0].imshow(sitk.GetImageFromArray(batch_data["image"]))
#   ax[0].set_title('Original')
#   ax[1].imshow(sitk.GetImageFromArray(batch_data["label"]))
#   ax[1].set_title('Transformed')
#   plt.show()
