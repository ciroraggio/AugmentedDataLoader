import SimpleITK as sitk
import numpy as np
from monai.transforms import (Compose, 
                              Rotate, 
                              RandStdShiftIntensity,
                              RandHistogramShift, 
                              RandFlip, 
                              Randomizable, 
                              RandScaleIntensity,
                              RandAdjustContrast,
                              RandGaussianSmooth,
                              RandZoom)

class RandomTransform(Randomizable):
    def __init__(self):
        super().__init__()

    def __call__(self, image):
        self.set_random_state(seed=np.random.randint(0, 10000))
        transform = Compose([
            # vanilla transforms
            ## intensity
            RandStdShiftIntensity(prob=0.5, factors=1),
            RandHistogramShift(prob=0.5),
            RandScaleIntensity(prob=0.5,  factors=1),
            RandAdjustContrast(prob=0.5),
            RandGaussianSmooth(prob=0.5),
            ## spatial
            RandFlip(prob=0.5),
            RandZoom(prob=0.5, min_zoom=0.8, max_zoom=1.1),
        ])
        return transform(image)

# An internal loop is used within the comprehensive list to apply NUM_TRANSFORMATIONS transformations to each image.
# The Sitk.GetarrayFromimage function is used to convert each image into an array number, which is then moved to the RandomTransform.
# Finally, the Sitk.GetimageFromarray function is used to convert the resulting array number in a Simpleitk image.
def augment(image_files, num_transformations=10):
    images = [sitk.ReadImage(file) for file in image_files]

    augmented_images = [
        sitk.GetImageFromArray(
            RandomTransform()(sitk.GetArrayFromImage(image))
        )
        for image in images
        for _ in range(num_transformations)
    ]

    return augmented_images