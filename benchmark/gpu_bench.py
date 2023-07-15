from monai.transforms import Rotate, Compose, Resize
from monai.data import ImageDataset
from AugmentedDataLoader import AugmentedDataLoader
import time
from memory_profiler import profile
import torch
import sys

bench_w_debug_log=open('gpu_with_debug_profile.log','w+')
bench_no_debug_log=open('gpu_no_debug_profile.log','w+')

@profile(stream=bench_no_debug_log)
def gpu_bench_no_debug(dataset, augmentation_transforms, batch_size, subset_len, debug_path, blocks):
    device = torch.device("cuda")
    data_loader = AugmentedDataLoader(dataset, augmentation_transforms, batch_size, subset_len, debug_path) # per debug
    for batch_data in data_loader.generate_batches():
        # inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
        # print(f"Batch data shape: {batch_data.shape}")
        inputs, labels = (batch_data[0], None) if len(batch_data) == 1 else batch_data
        inputs = inputs.to(device)
        labels = labels.to(device)
        blocks+=1
    print(f"Totale immagini: {num_patients}")
    print(f"Totale trasformazioni: {len(augmentation_transforms)}")
    print(f"Grandezza batch richiesta: {batch_size}")
    print(f"Blocchi ricevuti: {blocks}")

@profile(stream=bench_w_debug_log)
def gpu_bench_w_debug(dataset, augmentation_transforms, batch_size, subset_len, debug_path, blocks):
    device = torch.device("cuda")
    data_loader = AugmentedDataLoader(dataset, augmentation_transforms, batch_size, subset_len, debug_path) # per debug
    for batch_data in data_loader.generate_batches():
        # inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
        # print(f"Batch data shape: {batch_data.shape}")
        inputs, labels = (batch_data[0], None) if len(batch_data) == 1 else batch_data
        inputs = inputs.to(device)
        labels = labels.to(device)
        blocks+=1
    print(f"Totale immagini: {num_patients}")
    print(f"Totale trasformazioni: {len(augmentation_transforms)}")
    print(f"Grandezza batch richiesta: {batch_size}")
    print(f"Blocchi ricevuti: {blocks}")

if __name__ == '__main__':
    if torch.cuda.is_available():
        # ImageDataset params
        images_to_transform = [
            "./data/PDDCA-1.4.1_part1/0522c0001/img.nrrd",
            "./data/PDDCA-1.4.1_part1/0522c0002/img.nrrd",
            "./data/PDDCA-1.4.1_part1/0522c0003/img.nrrd",
            # "./data/PDDCA-1.4.1_part1/0522c0009/img.nrrd",
            # "./data/PDDCA-1.4.1_part1/0522c0013/img.nrrd",
        ]

        labels_to_transform = [
            "./data/PDDCA-1.4.1_part1/0522c0001/structures/Parotid_L.nrrd",
            "./data/PDDCA-1.4.1_part1/0522c0002/structures/Parotid_L.nrrd",
            "./data/PDDCA-1.4.1_part1/0522c0003/structures/Parotid_L.nrrd",
            # "./data/PDDCA-1.4.1_part1/0522c0009/structures/BrainStem.nrrd",
            # "./data/PDDCA-1.4.1_part1/0522c0013/structures/BrainStem.nrrd",
        ]
        each_image_trans = Compose([Resize([74,74,74])])

        # AugmentedDataLoader params
        augmentation_transforms = [
            Rotate(angle=35),
            Rotate(angle=61),
        ]
        batch_size = 2
        subset_len = 2
        num_patients = len(images_to_transform)
        dataset = ImageDataset(image_files=images_to_transform, labels=labels_to_transform, transform=each_image_trans)
        debug_path='./data/debug_path_test'
        blocks = 0

        start = time.time()
        gpu_bench_no_debug(dataset, augmentation_transforms, batch_size, subset_len, None, blocks)
        end = time.time()
        print(f"Tempo di esecuzione senza debug: {end-start} sec")

        start = time.time()
        gpu_bench_w_debug(dataset, augmentation_transforms, batch_size, subset_len, None, blocks)
        end = time.time()
        print(f"Tempo di esecuzione con debug: {end-start} sec")
    else:
        print("GPU non compatibile")
        sys.exit(-1)
