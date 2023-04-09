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
