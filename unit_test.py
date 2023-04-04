import unittest
from augmentation import augment

class TestAugment(unittest.TestCase):
    def test_no_num_transform(self):
        # If not specific a number of transformations, expect 10 transformations per image
        image_files = ['./data/MedNIST/HeadCT/000000.jpeg', './data/MedNIST/HeadCT/000001.jpeg']
        self.assertEqual(len(augment(image_files)), 20)

    def test_w_num_transform(self):
        # If not specific a number of transformations, expect 10 transformations per image
        image_files = ['./data/MedNIST/HeadCT/000000.jpeg', './data/MedNIST/HeadCT/000001.jpeg']
        self.assertEqual(len(augment(image_files, 5)), 10)
    