# MON_Augmentator
Medical image augmentation tool that can be integrated with Pytorch & MONAI.


## v.0.1a
- Defined a 'RandomTransform' class which implements the Randomizable interface and a function 'augment' which allows to augment images by randomly applying a series of transformations.


### Other infos
To run tests: 
 - python -m unittest (all tests)
 - python -m unittest file_name.TestClassName 
 - python -m unittest file_name.TestClassName.test_name 