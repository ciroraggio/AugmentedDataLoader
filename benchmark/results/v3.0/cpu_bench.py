from monai.transforms import Rotate, Compose, Resize, EnsureChannelFirst
from monai.data import ImageDataset
import timeit
from memory_profiler import profile
import monai.transforms as monai_transf
import cProfile
from loaders.AugmentedDataLoader import AugmentedDataLoader


bench_w_debug_0_trans_prof = cProfile.Profile()
bench_w_debug_5_trans_prof = cProfile.Profile()
bench_w_debug_10_trans_prof = cProfile.Profile()

bench_no_debug_0_trans_prof = cProfile.Profile()
bench_no_debug_5_trans_prof = cProfile.Profile()
bench_no_debug_10_trans_prof = cProfile.Profile()

# no debug
def cpu_bench_no_debug_0_transf(dataset, augmentation_transforms, batch_size, subset_len):
    bench_no_debug_0_trans_prof.enable()
    augmented_data_loader = AugmentedDataLoader(dataset, augmentation_transforms, batch_size, subset_len, shuffle_mode="full")
    for img_batch, seg_batch in augmented_data_loader:
        continue
    bench_no_debug_0_trans_prof.disable()

def cpu_bench_no_debug_5_transf(dataset, augmentation_transforms, batch_size, subset_len):
    bench_no_debug_5_trans_prof.enable()
    augmented_data_loader = AugmentedDataLoader(dataset, augmentation_transforms, batch_size, subset_len, shuffle_mode="full")
    for img_batch, seg_batch in augmented_data_loader:
        continue
    bench_no_debug_5_trans_prof.disable()

def cpu_bench_no_debug_10_transf(dataset, augmentation_transforms, batch_size, subset_len):
    bench_no_debug_10_trans_prof.enable()
    augmented_data_loader = AugmentedDataLoader(dataset, augmentation_transforms, batch_size, subset_len, shuffle_mode="full")
    for img_batch, seg_batch in augmented_data_loader:
        continue
    bench_no_debug_10_trans_prof.disable()

# with debug
def cpu_bench_w_debug_0_transf(dataset, augmentation_transforms, batch_size, subset_len):
    bench_w_debug_0_trans_prof.enable()
    augmented_data_loader = AugmentedDataLoader(dataset, augmentation_transforms, batch_size, subset_len, shuffle_mode="full")
    for img_batch, seg_batch in augmented_data_loader:
        continue
    bench_w_debug_0_trans_prof.disable()

def cpu_bench_w_debug_5_transf(dataset, augmentation_transforms, batch_size, subset_len):
    bench_w_debug_5_trans_prof.enable()
    augmented_data_loader = AugmentedDataLoader(dataset, augmentation_transforms, batch_size, subset_len, shuffle_mode="full")
    for img_batch, seg_batch in augmented_data_loader:
        continue
    bench_w_debug_5_trans_prof.disable()

def cpu_bench_w_debug_10_transf(dataset, augmentation_transforms, batch_size, subset_len):
    bench_w_debug_10_trans_prof.enable()
    augmented_data_loader = AugmentedDataLoader(dataset, augmentation_transforms, batch_size, subset_len, shuffle_mode="full")
    for img_batch, seg_batch in augmented_data_loader:
        continue
    bench_w_debug_10_trans_prof.disable()



if __name__ == '__main__':

    print("++++++++++++++++ AugmentedDataLoader CPU Benchmark Started ++++++++++++++++")
    print("++++++++++++++++++++++++++++++ Please wait... +++++++++++++++++++++++++++++")
    # ImageDataset params
    images_to_transform = [
        "/home/jovyan/AugmentedDataLoader/benchmark/data/PDDCA-1.4.1_part1/0522c0013/img.nrrd",
        "/home/jovyan/AugmentedDataLoader/benchmark/data/PDDCA-1.4.1_part1/0522c0014/img.nrrd",
        "/home/jovyan/AugmentedDataLoader/benchmark/data/PDDCA-1.4.1_part1/0522c0017/img.nrrd",
        "/home/jovyan/AugmentedDataLoader/benchmark/data/PDDCA-1.4.1_part1/0522c0057/img.nrrd",
        "/home/jovyan/AugmentedDataLoader/benchmark/data/PDDCA-1.4.1_part1/0522c0070/img.nrrd",
        "/home/jovyan/AugmentedDataLoader/benchmark/data/PDDCA-1.4.1_part1/0522c0077/img.nrrd",
        "/home/jovyan/AugmentedDataLoader/benchmark/data/PDDCA-1.4.1_part1/0522c0079/img.nrrd",
        "/home/jovyan/AugmentedDataLoader/benchmark/data/PDDCA-1.4.1_part1/0522c0081/img.nrrd",
        "/home/jovyan/AugmentedDataLoader/benchmark/data/PDDCA-1.4.1_part1/0522c0125/img.nrrd",
        "/home/jovyan/AugmentedDataLoader/benchmark/data/PDDCA-1.4.1_part1/0522c0132/img.nrrd",
        "/home/jovyan/AugmentedDataLoader/benchmark/data/PDDCA-1.4.1_part1/0522c0147/img.nrrd",
        "/home/jovyan/AugmentedDataLoader/benchmark/data/PDDCA-1.4.1_part1/0522c0149/img.nrrd",
    ]

    seg_to_transform = [
        "/home/jovyan/AugmentedDataLoader/benchmark/data/PDDCA-1.4.1_part1/0522c0013/structures/BrainStem.nrrd",
        "/home/jovyan/AugmentedDataLoader/benchmark/data/PDDCA-1.4.1_part1/0522c0014/structures/BrainStem.nrrd",
        "/home/jovyan/AugmentedDataLoader/benchmark/data/PDDCA-1.4.1_part1/0522c0017/structures/BrainStem.nrrd",
        "/home/jovyan/AugmentedDataLoader/benchmark/data/PDDCA-1.4.1_part1/0522c0057/structures/BrainStem.nrrd",
        "/home/jovyan/AugmentedDataLoader/benchmark/data/PDDCA-1.4.1_part1/0522c0070/structures/BrainStem.nrrd",
        "/home/jovyan/AugmentedDataLoader/benchmark/data/PDDCA-1.4.1_part1/0522c0077/structures/BrainStem.nrrd",
        "/home/jovyan/AugmentedDataLoader/benchmark/data/PDDCA-1.4.1_part1/0522c0079/structures/BrainStem.nrrd",
        "/home/jovyan/AugmentedDataLoader/benchmark/data/PDDCA-1.4.1_part1/0522c0081/structures/BrainStem.nrrd",
        "/home/jovyan/AugmentedDataLoader/benchmark/data/PDDCA-1.4.1_part1/0522c0125/structures/BrainStem.nrrd",
        "/home/jovyan/AugmentedDataLoader/benchmark/data/PDDCA-1.4.1_part1/0522c0132/structures/BrainStem.nrrd",
        "/home/jovyan/AugmentedDataLoader/benchmark/data/PDDCA-1.4.1_part1/0522c0147/structures/BrainStem.nrrd",
        "/home/jovyan/AugmentedDataLoader/benchmark/data/PDDCA-1.4.1_part1/0522c0149/structures/BrainStem.nrrd",
    ]

    labels=[1,0]
    each_image_trans = Compose([
        EnsureChannelFirst(channel_dim="no_channel"),
        Resize(spatial_size=(178, 248, 248), mode='trilinear')
    ])

    # AugmentedDataLoader params
    augmentation_transforms_5 = [     
        monai_transf.Flip(spatial_axis=1),
        monai_transf.RandAffine(prob=1, rotate_range=[-4, 4], scale_range=[-0.4,0.4]),
        monai_transf.ScaleIntensityRange(a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
        monai_transf.Flip(spatial_axis=0),
        monai_transf.Rotate(angle=[-0.53,-0.53,-0.53]),
    ]

    augmentation_transforms_10 = [
        monai_transf.Flip(spatial_axis=1),
        monai_transf.RandAffine(prob=1, rotate_range=[-4, 4], scale_range=[-0.4,0.4]),
        monai_transf.ScaleIntensityRange(a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
        monai_transf.Flip(spatial_axis=0),
        monai_transf.Rotate(angle=[-0.53,-0.53,-0.53]),
        monai_transf.Rotate(angle=[-0.15,-0.15,-0.15]),
        monai_transf.Rotate(angle=[0.15,0.15,0.15]),
        monai_transf.Rotate(angle=[0.1,0.1,0.1]),
        monai_transf.Rotate(angle=[1,1,1]),
        monai_transf.Rotate(angle=[1.3,1.3,1.3]),
    ]

    batch_size = 3
    subset_len = 3
    debug_path='./benchmark/data/debug_path_test'
    time_iteration = 2


    dataset = ImageDataset(image_files=images_to_transform, seg_files=seg_to_transform, transform=each_image_trans, seg_transform=each_image_trans) # test solo segmentazioni
    # no debug    
    print("++++++++++++++++ Start 5 transform (no debug) benchmark ++++++++++++++++")
    five_trans_time_no_debug = timeit.timeit(stmt='cpu_bench_no_debug_5_transf(dataset, augmentation_transforms_5, batch_size, subset_len)', globals=globals(), number=time_iteration)
    print(f"{bench_no_debug_5_trans_prof.print_stats()}")
    print(f"Execution time: {five_trans_time_no_debug / time_iteration} seconds")
    print("++++++++++++++++++++++++++++++++ Finish ++++++++++++++++++++++++++++++++")
    
    print("++++++++++++++++ Start 10 transform (no debug) benchmark +++++++++++++++")
    ten_trans_time_no_debug = timeit.timeit(stmt='cpu_bench_no_debug_10_transf(dataset, augmentation_transforms_10, batch_size, subset_len)', globals=globals(), number=time_iteration)
    print(f"{bench_no_debug_10_trans_prof.print_stats()}")
    print(f"Execution time is: {ten_trans_time_no_debug / time_iteration} seconds")
    print("++++++++++++++++++++++++++++++++ Finish ++++++++++++++++++++++++++++++++")

    # with debug
    print("+++++++++++++++ Start 5 transform (with debug) benchmark +++++++++++++++")
    five_trans_time_debug = timeit.timeit(stmt='cpu_bench_w_debug_5_transf(dataset, augmentation_transforms_5, batch_size, subset_len)', globals=globals(), number=time_iteration)
    print(f"{bench_w_debug_5_trans_prof.print_stats()}")
    print(f"Execution time is: {five_trans_time_debug / time_iteration} seconds")
    print("++++++++++++++++++++++++++++++++ Finish ++++++++++++++++++++++++++++++++")

    print("+++++++++++++++ Start 10 transform (with debug) benchmark ++++++++++++++")
    ten_trans_time_debug = timeit.timeit(stmt='cpu_bench_w_debug_10_transf(dataset, augmentation_transforms_10, batch_size, subset_len)', globals=globals(), number=time_iteration)
    print(f"{bench_w_debug_10_trans_prof.print_stats()}")
    print(f"Execution time is: {ten_trans_time_debug / time_iteration} seconds")
    print("++++++++++++++++++++++++++++++++ Finish ++++++++++++++++++++++++++++++++")

    print("++++++++++++++++ Finish! ++++++++++++++++")