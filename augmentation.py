import SimpleITK as sitk
from utils import (drop_duplicate_images, check_paths_exists, string_equals)
from augmentation_types import RANDOM_AUGMENTATION
from RandomTransform import RandomTransform

# This is the function that the user will call from the outside
def augment(image_paths, augmentation_type=RANDOM_AUGMENTATION, num_transformations=10):
    paths_exists, wrong_path = check_paths_exists(image_paths)
    if(paths_exists):
        if(string_equals(augmentation_type, RANDOM_AUGMENTATION)):
            return random_augment(image_paths, num_transformations)
        else:
            return f'AugmentationType {augmentation_type} not supported'
    else:
        return f'{wrong_path} not founded!'
    
# "random_augment" function
# An internal loop is used within the comprehensive list to apply NUM_TRANSFORMATIONS transformations to each image.
# The Sitk.GetArrayFromImage function is used to convert each image into an array number, which is then moved to the RandomTransform.
# Finally, the Sitk.GetImageFromArray function is used to convert the resulting array number in a Simpleitk image.
def random_augment(image_files, num_transformations=10):
    images = [sitk.ReadImage(file) for file in image_files]

    augmented_images = [
        sitk.GetImageFromArray(
            RandomTransform()(sitk.GetArrayFromImage(image))
        )
        for image in images
        for _ in range(num_transformations)
    ]

    return drop_duplicate_images(augmented_images)