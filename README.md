# mon-augmentator
Medical image augmentation tool that can be integrated with Pytorch & MONAI.

## v0.3a
Completely revised the augmentation logic:
- Added methods for in-memory augmentation by extending the MONAI methods
- Removed old code
- Updated requirements
- Updated version and README
  
## v0.2a
- Redefined the 'augment' function and added the 'random_augment' function
- Added utils functions for paths existence, string equality etc.
- Added supported augmentation types
- Added more unit tests
- Code refactoring
  
## v0.1a
- Defined a 'RandomTransform' class which implements the Randomizable interface and a function 'augment' which allows to augment images by randomly applying a series of transformations.


### Other infos
To run tests: 
 - python -m unittest (all tests)
 - python -m unittest file_name.TestClassName 
 - python -m unittest file_name.TestClassName.test_name 