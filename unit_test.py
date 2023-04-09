import unittest, SimpleITK as sitk
from augmentation import augment
from utils import drop_duplicate_images, check_paths_exists, is_array, string_equals
from augmentation_types import RANDOM_AUGMENTATION
class TestAugment(unittest.TestCase):
    def test_no_num_transform_random(self):
        # If not specific a number of transformations, expect 10 transformations per image
        image_files = ['./data/MedNIST/HeadCT/000000.jpeg', './data/MedNIST/HeadCT/000001.jpeg']
        self.assertEqual(len(augment(image_files, RANDOM_AUGMENTATION)), 20)

    def test_w_num_transform_random(self):
        image_files = ['./data/MedNIST/HeadCT/000000.jpeg', './data/MedNIST/HeadCT/000001.jpeg']
        self.assertEqual(len(augment(image_files, RANDOM_AUGMENTATION, 5)), 10)

class TestUtils(unittest.TestCase):
    def test_drop_duplicate_images(self):
        original_0 = sitk.GetArrayFromImage(sitk.ReadImage("./data/MedNIST/AbdomenCT/000000.jpeg"))
        original_1 = sitk.GetArrayFromImage(sitk.ReadImage("./data/MedNIST/AbdomenCT/000001.jpeg"))
        np_images = [original_0,original_0,original_1,original_1]
        self.assertEqual(len(drop_duplicate_images(np_images)), 2)

    def test_check_paths_exists(self):
        test_true, wrong_path = check_paths_exists("./data/MedNIST/AbdomenCT/000001.jpeg")
        self.assertEqual(test_true, True)
        self.assertEqual(wrong_path, None)
        
        test_false, wrong_path = check_paths_exists("./data/MedNIST/000001.jpeg")
        self.assertEqual(test_false, False)
        self.assertEqual(wrong_path, "./data/MedNIST/000001.jpeg")

    def test_is_array(self):
        self.assertEqual(is_array([1,2,3]), True)
        self.assertEqual(is_array("This is a string"), False)

    def test_string_equals(self):
        self.assertEqual(string_equals('Foo', 'FOO'), True)
        self.assertEqual(string_equals('Foo', 'Bar'), False)