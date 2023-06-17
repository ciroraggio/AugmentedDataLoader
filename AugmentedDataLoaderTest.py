import unittest
import random
from monai.transforms import Rotate, Compose, Resize
from AugmentedDataLoader import AugmentedDataLoader
from monai.data import ImageDataset

class AugmentedDataLoaderTest(unittest.TestCase):
    def test_generate_batches(self):
        # Define test inputs
        images_to_transform = [
            "./data/PDDCA-1.4.1_part1/0522c0001/img.nrrd",
            "./data/PDDCA-1.4.1_part1/0522c0002/img.nrrd",
            "./data/PDDCA-1.4.1_part1/0522c0003/img.nrrd",
            "./data/PDDCA-1.4.1_part1/0522c0009/img.nrrd",
            "./data/PDDCA-1.4.1_part1/0522c0013/img.nrrd",
        ]

        labels_to_transform = [
            "./data/PDDCA-1.4.1_part1/0522c0001/structures/BrainStem.nrrd",
            "./data/PDDCA-1.4.1_part1/0522c0002/structures/BrainStem.nrrd",
            "./data/PDDCA-1.4.1_part1/0522c0003/structures/BrainStem.nrrd",
            "./data/PDDCA-1.4.1_part1/0522c0009/structures/BrainStem.nrrd",
            "./data/PDDCA-1.4.1_part1/0522c0013/structures/BrainStem.nrrd",
        ]
        each_image_trans = Compose([Resize(74,74,74)])

        # AugmentedDataLoader params
        augmentation_transforms = [
            Rotate(angle=35),
            Rotate(angle=61),
        ]
        batch_size = 2
        subset_len=20
        num_patients = len(images_to_transform)
        dataset = ImageDataset(image_files=images_to_transform, labels=labels_to_transform, transform=each_image_trans)
        data_loader = AugmentedDataLoader(dataset, augmentation_transforms, batch_size, subset_len)
        batches = list(data_loader)
        
        # Assert conditions
        # The final batch with fewer patients should be accounted if the total number of patients is not an integer multiple of the batch size.
        if num_patients % batch_size != 0:
            expected_num_batches += 1
        self.assertEqual(len(batches), expected_num_batches)  # Check the number of batches generated

        # Check the batch size
        for batch in batches:
            self.assertLessEqual(batch.shape[0], batch_size)  # Each batch should have at most batch_size number of images
        
        # Check that all patients are included in the generated batches
        all_images = []
        for batch in batches:
            all_images.extend(batch)
        expected_num_images = num_patients * (len(augmentation_transforms) + 1)
        self.assertEqual(len(all_images), expected_num_images)  # Each patient should have (M + 1) images (original + augmented)


if __name__ == '__main__':
    unittest.main()
