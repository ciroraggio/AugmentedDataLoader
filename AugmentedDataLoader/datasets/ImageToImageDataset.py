from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from monai.config import DtypeLike
from monai.data.image_reader import ImageReader
from monai.transforms import LoadImage, Randomizable, apply_transform
from monai.utils import MAX_SEED, get_seed
from typing import Optional

class ImageToImageDataset(Dataset, Randomizable):
    """
    Loads image_one/image_two/segmentation triplet of files from the given filename lists. Transformations can be specified
    for the image_one and image_two and segmentation arrays separately.
    This extends the plain ImageDataset from MONAI.
    """

    def __init__(
        self,
        first_type_image_files: Sequence[str],
        second_type_image_files: Sequence[str],
        seg_files: Sequence[str] | None = None,
        labels: Sequence[float] | None = None,
        first_type_image_transforms: Callable | None = None,
        second_type_image_transforms: Callable | None = None,
        seg_transform: Callable | None = None,
        label_transform: Callable | None = None,
        dtype: DtypeLike = np.float32,
        reader: ImageReader | str | None = None,
        device: str = "cpu",
        *args,
        **kwargs,
    ) -> None:
        if seg_files is not None and (len(first_type_image_files) != len(seg_files) or len(first_type_image_files) != len(seg_files)):
            raise ValueError(
                "Must have same the number of segmentation as image files: "
                f"first type of images={len(first_type_image_files)}, second type of images={len(first_type_image_files)}, segmentations={len(seg_files)}."
            )

        self.first_type_image_files = first_type_image_files
        self.second_type_image_files = second_type_image_files
        self.seg_files = seg_files
        self.labels = labels
        self.first_type_image_transforms = first_type_image_transforms
        self.second_type_image_transforms = second_type_image_transforms
        self.seg_transform = seg_transform
        self.label_transform = label_transform
        self.loader = LoadImage(reader, True, dtype, *args, **kwargs)
        self.device = device
        self.set_random_state(seed=get_seed())
        self._seed = 0  # transform synchronization seed

    def __len__(self) -> int:
        return len(self.first_type_image_files)

    def randomize(self, data: Any | None = None) -> None:
        self._seed = self.R.randint(MAX_SEED, dtype="uint32")

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        self.randomize()
        seg, label = None, None
        try:
            # load data
            first_img = self.loader(self.first_type_image_files[index]).to(self.device)
            second_img = self.loader(self.second_type_image_files[index]).to(self.device)
            if self.seg_files is not None:
                seg = self.loader(self.seg_files[index]).to(self.device)

            # apply the transforms
            if self.first_type_image_transforms is not None:
                first_img = apply_transform(self.first_type_image_transforms, first_img, map_items=False)
                
            if self.second_type_image_transforms is not None:
                second_img = apply_transform(self.second_type_image_transforms, second_img, map_items=False)

            if self.seg_files is not None and self.seg_transform is not None:
                seg = apply_transform(self.seg_transform, seg, map_items=False)

            if self.labels is not None:
                label = self.labels[index]
                if self.label_transform is not None:
                    label = apply_transform(self.label_transform, label, map_items=False)  # type: ignore

            data = [first_img, second_img]
            if seg is not None:
                data.append(seg)
            if label is not None:
                data.append(label)
            if len(data) == 1:
                return data[0]

            # use tuple instead of list as the default collate_fn callback of MONAI DataLoader flattens nested lists
            return tuple(data)

        except EOFError as e:
            print(f"EOFError: {e}. Skipping to the next image.")
            return self.__getitem__((index + 1) % len(self))