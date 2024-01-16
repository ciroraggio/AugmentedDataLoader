import unittest
from monai.transforms import Rotate, Compose, Resize
from loaders.AugmentedDataLoader import AugmentedDataLoader
from monai.data import ImageDataset
import torch
class AugmentedDataLoaderTest(unittest.TestCase):
    
    def setUp(self):
        self.images_to_transform = [
                "./benchmark/data/PDDCA-1.4.1_part1/0522c0001/img.nrrd",
                "./benchmark/data/PDDCA-1.4.1_part1/0522c0002/img.nrrd",
                "./benchmark/data/PDDCA-1.4.1_part1/0522c0003/img.nrrd",
                "./benchmark/data/PDDCA-1.4.1_part1/0522c0009/img.nrrd",
                "./benchmark/data/PDDCA-1.4.1_part1/0522c0013/img.nrrd",
            ]

        self.seg_to_transform = [
                "./benchmark/data/PDDCA-1.4.1_part1/0522c0001/structures/BrainStem.nrrd",
                "./benchmark/data/PDDCA-1.4.1_part1/0522c0002/structures/BrainStem.nrrd",
                "./benchmark/data/PDDCA-1.4.1_part1/0522c0003/structures/BrainStem.nrrd",
                "./benchmark/data/PDDCA-1.4.1_part1/0522c0009/structures/BrainStem.nrrd",
                "./benchmark/data/PDDCA-1.4.1_part1/0522c0013/structures/BrainStem.nrrd",
            ]
        self.each_image_trans = Compose([Resize([25,25,1])])

            # AugmentedDataLoader params
        self.M = [
                Rotate(angle=[0.4, 0.4,0.4]),
                Rotate(angle=[0.2, 0.2,0.2]),
            ]
        self.K = 2 # batch dimension
        self.J = 2 # subset dimension
        self.N = len(self.images_to_transform)
        self.dataset = ImageDataset(image_files=self.images_to_transform, seg_files=self.seg_to_transform, transform=self.each_image_trans, seg_transform=self.each_image_trans)
        self.data_loader = AugmentedDataLoader(self.dataset, self.M, self.K, self.J)
        
    def test_num_blocks(self):
        """ 
        Test 1. The number of blocks received must be equal to (N + (N*len(M))) / K
        Example:
            If we have 2 images and 2 transformations we get 4 augmented + 2 non-augmented images, so 6 images.
            If we declare batch size as 3, we should get 2 blocks of 3 images -> NB = (N + (N*len(M))) / K = (2 + 2*2) / 3 = 2
        """
        # if N % K != 0 one image remains in the last block so I get one additional block
        expected_num_blocks = int((self.N + (self.N * len(self.M))) / self.K) + 1 if self.N % self.K != 0 else int((self.N + (self.N * len(self.M))) / self.K)
        blocks_received = 0
        for batch in self.data_loader:
            blocks_received +=1
        self.assertEqual(blocks_received, expected_num_blocks, "Test 1")

    def test_batch_size(self):
            """
            Test 2. Batch size must be equal to K 
            """
            for batch in self.data_loader:
                self.assertLessEqual(len(batch), self.K, "Test 2") 

    def test_total_images(self):
            """
            Test 3. The total number of images obtained by the augmentation must be equal to N + (N *len(M)).
            Example:
                If we have 2 images and 2 transformations we get 4 augmented + 2 non-augmented images, so 6 images.
            """
            img_received = 0
            exp_img = self.N + (self.N * len(self.M)) # same for segmentations
            for batch in self.data_loader:
                img_received += len(batch[0]) # batch[0] contains the images
            self.assertLessEqual(exp_img, img_received, "Test 3") 
            
    def test_is_float(self):
            for batch in self.data_loader:
                images = batch[0]
                segs = batch[1]
                for image in images:
                    self.assertTrue(torch.is_floating_point(image), "Test 4 - Images")
                    self.assertTrue(torch.is_floating_point(segs), "Test 4 - Segmentations")
            


if __name__ == '__main__':
    unittest.main()
