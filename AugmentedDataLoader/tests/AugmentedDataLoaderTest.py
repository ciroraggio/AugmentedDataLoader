import unittest
import torch
from monai.transforms import RandRotate, Compose, Resize, Flip
from monai.data import ImageDataset
from AugmentedDataLoader.loaders import AugmentedDataLoader

class AugmentedDataLoaderTest(unittest.TestCase):
    
    def setUp(self):
        self.images_to_transform = [
            "./benchmark/data/PDDCA-1.4.1_part1/0522c0013/img.nrrd",
            "./benchmark/data/PDDCA-1.4.1_part1/0522c0014/img.nrrd",
            "./benchmark/data/PDDCA-1.4.1_part1/0522c0017/img.nrrd",
            "./benchmark/data/PDDCA-1.4.1_part1/0522c0057/img.nrrd",
        ]

        self.seg_to_transform = [
            "./benchmark/data/PDDCA-1.4.1_part1/0522c0013/structures/BrainStem.nrrd",
            "./benchmark/data/PDDCA-1.4.1_part1/0522c0014/structures/BrainStem.nrrd",
            "./benchmark/data/PDDCA-1.4.1_part1/0522c0017/structures/BrainStem.nrrd",
            "./benchmark/data/PDDCA-1.4.1_part1/0522c0057/structures/BrainStem.nrrd",
        ]

        self.labels_to_transform = [i for i in range(len(self.images_to_transform))]

        self.each_image_trans = Compose([Resize([178, 64, 64])])

        self.M = [
            Flip(spatial_axis=-1),
            RandRotate(range_x=[-0.5, 0.5], range_z=[-0.5, 0.5], range_y=[-0.5, 0.5], prob=1, keep_size=True),
        ]
        
        self.K = 3  # batch dimension
        self.J = 2  # subset dimension
        self.N = len(self.images_to_transform)
        self.device=torch.device(0)
        self.dataset = ImageDataset(
            image_files=self.images_to_transform,
            seg_files=self.seg_to_transform,
            transform=self.each_image_trans,
            seg_transform=self.each_image_trans
        )

        self.data_loader = AugmentedDataLoader(
            dataset=self.dataset,
            augmentation_transforms=self.M,
            batch_size=self.K,
            subset_len=self.J,
            transformation_device=torch.device(0),
            return_on_device=torch.device(0)
        )

        self.dataset_with_labels = ImageDataset(image_files=self.images_to_transform, labels=self.labels_to_transform, transform=self.each_image_trans)
        self.data_loader_with_labels = AugmentedDataLoader(self.dataset_with_labels, self.M, self.K, self.J, transformation_device=torch.device(0), return_on_device=torch.device(0))


    def test_num_batch(self):
        """
        Test 1. The number of batches received must be equal to (N + (N*len(M))) / K
        Example:
            If we have 2 images and 2 transformations we get 4 augmented + 2 non-augmented images, so 6 images.
            If we declare batch size as 3, we should get 2 batches of 3 images -> NB = (N + (N*len(M))) / K = (2 + 2*2) / 3 = 2
        """
        expected_num_batches = (self.N + (self.N * len(self.M))) // self.K
        
        if (self.N + (self.N * len(self.M))) % self.K != 0:
            expected_num_batches += 1

        batches_received = sum(1 for _ in self.data_loader)
        self.assertEqual(batches_received, expected_num_batches, f"Test 1 {self._testMethodName} failed.")

    def test_batch_size(self):
        """
        Test 2. Batch size must be equal to K 
        """
        for batch in self.data_loader:
            self.assertLessEqual(len(batch[0]), self.K, f"Test 2 {self._testMethodName} failed.")

    def test_total_images(self):
        """
        Test 3. The total number of images obtained by the augmentation must be equal to N (original images) + (N *len(M)) (augmented images).
        Example:
            If we have 2 images and 2 transformations we get 4 augmented + 2 non-augmented images, so 6 images.
        """
        total_images_received = sum(len(batch[0]) for batch in self.data_loader)
        expected_total_images = self.N + (self.N * len(self.M))
        self.assertEqual(total_images_received, expected_total_images, f"Test 3 {self._testMethodName} failed.")

    def test_is_float(self):
        """
        Test 4. Ensure all images and segmentations in batches are of type float.
        """
        for image_batch, seg_batch in self.data_loader:
            for image, seg in zip(image_batch, seg_batch):
                self.assertTrue(torch.is_floating_point(image), f"Test 4 {self._testMethodName} - Images failed.")
                self.assertTrue(torch.is_floating_point(seg), f"Test 4 {self._testMethodName} - Segmentations failed.")

    def test_labels_are_1d_tensor(self):
        """
        Test 5. Verify that labels are a 1D tensor if the dataset contains labels instead of segmentations.
        """
        for image_batch, label_batch in self.data_loader_with_labels:
            self.assertEqual(len(label_batch.size()), 1, f"Test 5 {self._testMethodName} - Labels are 1D tensor failed")
    
    def test_return_device(self):
        """
        Test 6. Verify that the tensors are on the selected device.
        """
        for image_batch, seg_batch in self.data_loader:
            self.assertEqual(image_batch.device, torch.device(0), f"Test 6 {self._testMethodName} - Image batch on return_device failed")
            self.assertEqual(seg_batch.device, torch.device(0), f"Test 6 {self._testMethodName} - Image batch on return_device failed")
        
        for image_batch, label_batch in self.data_loader_with_labels:
            self.assertEqual(label_batch.device, torch.device(0), f"Test 6 {self._testMethodName} - Labels batch on return_device failed")

if __name__ == '__main__':
    unittest.main()
