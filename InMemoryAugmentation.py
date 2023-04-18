from torch.utils.data import DataLoader
from torch.utils.data import SequentialSampler
from monai.data import SmartCacheDataset
from monai.transforms import (
    Compose,
    LoadImaged,
    Resized,
    EnsureTyped,
    ToDeviced,
    ToTensor
)
import torch


class MySmartCacheDataset(SmartCacheDataset):
    def __getitem__(self, index):
        data = self.data[index]
        image = data["img"]
        seg = data["seg"]
        if self.transform is not None:
            image = self.transform(image)
            seg = self.transform(seg)
        return {"img": ToTensor()(image), "seg": ToTensor()(seg)}


class TransformableBatchSampler:
    def __init__(self, sampler, batch_size, drop_last, transform=None):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.transform = transform
        self._iter = None

    def set_transform(self, transform):
        print(f"Setting batch transform: {transform}")
        self.transform = transform

    """
    The __iter__ method of the class loops over the items in the sampler and adds them to a batch until the batch size is reached. 
    When the batch size is reached, the transform function (if it exists) is applied to each item in the batch, and the batch is yielded. 
    If drop_last is True, any remaining items in the sampler that do not fit into a full batch are ignored. 
    If drop_last is False, a final batch is yielded with the remaining items, and the transform function (if it exists) is applied to each item in the batch.
    """

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                if self.transform is not None:
                    print(
                        f"Finded trasform: {self.transform}\nwith sampler element: {self.sampler}\nand idx: {idx}"
                    )
                    batch = [
                        self.transform(self.sampler[indx])
                        for indx, _ in enumerate(self.sampler)
                    ]
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            if self.transform is not None:
                print(
                    f"Finded trasform: {self.transform}\nwith sampler element: {self.sampler}\nand idx: {idx}"
                )
                batch = [
                    self.transform(self.sampler[indx])
                    for indx, _ in enumerate(self.sampler)
                ]
            yield batch

    def _reset_iter(self):
        self._iter = None

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size


class TransformableDataLoader(DataLoader):
    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        collate_fn=None,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
        multiprocessing_context=None,
        transform=None,
    ):
        if batch_sampler is None:
            if sampler is None:
                sampler = SequentialSampler(dataset)
            batch_sampler = TransformableBatchSampler(
                sampler, batch_size=batch_size, drop_last=drop_last, transform=transform
            )
        super(TransformableDataLoader, self).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
            multiprocessing_context=multiprocessing_context,
        )

        self.transform = transform

    def set_transform(self, transform):
        self.transform = transform

    def __iter__(self):
        self.batch_sampler.set_transform(
            self.transform
        )  # set the transform for the batch_sampler
        self.batch_sampler._reset_iter()  # reset the iterator of the batch_sampler
        return iter(self.batch_sampler)  # create and return iterator on the batch_sampler


def get_transforms(transform):
    if torch.cuda.is_available():
        return Compose(
            [
                LoadImaged(
                    keys=["img", "seg"], image_only=True, ensure_channel_first=True
                ),
                EnsureTyped(
                    keys=["img", "seg"],
                    data_type="tensor",
                    device="cuda:0",
                    track_meta=False,
                ),
                ToDeviced(keys=["img", "seg"], device="cuda:0"),
                Resized(keys=["img", "seg"], spatial_size=(512, 512, 130)),
                transform,
            ]
        )
    return Compose(
        [
            LoadImaged(keys=["img", "seg"], image_only=True, ensure_channel_first=True),
            EnsureTyped(keys=["img", "seg"], data_type="tensor", track_meta=False),
            Resized(keys=["img", "seg"], spatial_size=(512, 512, 130)),
            transform,
        ]
    )


def get_dict_list(images, segmentations):
    return [{"img": image, "seg": seg} for image, seg in zip(images, segmentations)]


def augment(images, segmentations, transforms):
    augmented_data = []
    data_dict_list = get_dict_list(images, segmentations)

    cache_augm_ds = SmartCacheDataset(
        data=data_dict_list,
        replace_rate=0.2,
        cache_num=15,
        num_init_workers=2,
        num_replace_workers=2,
    )

    batch_sampler = TransformableBatchSampler(
        cache_augm_ds,
        batch_size=3,
        drop_last=True,
    )

    data_loader = TransformableDataLoader(
        cache_augm_ds,
        batch_sampler=batch_sampler,
        num_workers=8,
    )

    # iterazione dataloader con trasformazioni diverse
    for transform in transforms:
        print(f"Selected transform: {transform}")
        temp_transf = get_transforms(transform)
        print(f"Changing transform: {temp_transf}")
        data_loader.set_transform(
            temp_transf
        )  # cambia la trasformazione del batch_sampler
        print(f"Dataloader transform: {data_loader.transform}")

        print(f"Creating an iterable data loader")
        iter_loader = iter(data_loader)  # crea un nuovo iteratore sul dataloader

        for batch in iter_loader:
            for _, a_batch in enumerate(batch):
                print(f"Start augmentation for this batch: {a_batch}")
                augmented_data.append(a_batch)

    return augmented_data
