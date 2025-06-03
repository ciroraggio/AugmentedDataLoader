import matplotlib.pyplot as plt
import os
import torch

def save_subplot(batch, output_path, batch_num):
    """
    Extracts center slices from a batch of images and saves them to a specified path.

    Parameters:
    - batch: torch.tensor [batch, channels, num_slices, H, W]
    - output_path: str, path where to save the extracted slice
    - batch_num: int, batch number to distinguish it in file names
    """
    B = batch.size(0)
    if len(batch.shape) == 5:
        C, S, H, W = batch.shape[1:]
        central_slice_idx = S // 2
    else:
        C, H, W = batch.shape[1:]
        central_slice_idx = 0  # 2D case

    for i in range(B):
        if len(batch.shape) == 5:
            central_slice = batch[i, :, central_slice_idx, :, :]
        else:
            central_slice = batch[i]

        if C == 1:
            central_slice = central_slice.squeeze(0)
        
        plt.imsave(os.path.join(output_path, f'batch{batch_num}_image_{i}_slice_{central_slice_idx}.png'), central_slice.cpu(), cmap='gray')
        plt.close()

def save_subplot_i2i(batch_first_images, batch_second_images, output_path, batch_num):
    B = batch_first_images.size(0)
    if len(batch_first_images.shape) == 5:
        C, S, H, W = batch_first_images.shape[1:]
        central_slice_idx = S // 2
    else:
        C, H, W = batch_first_images.shape[1:]
        central_slice_idx = 0  # 2D case

    fig, axes = plt.subplots(1, 2)
    for i in range(B):
        if len(batch_first_images.shape) == 5:
            central_slice_first = batch_first_images[i, :, central_slice_idx, :, :]
            central_slice_second = batch_second_images[i, :, central_slice_idx, :, :]
        else:
            central_slice_first = batch_first_images[i]
            central_slice_second = batch_second_images[i]

        if C == 1:
            central_slice_first = central_slice_first.squeeze(0)
            central_slice_second = central_slice_second.squeeze(0)


        axes[0].imshow(central_slice_first.cpu(), cmap='gray')
        axes[0].set_title('First type img')
        axes[0].axis('off')

        axes[1].imshow(central_slice_second.cpu(), cmap='gray')
        axes[1].set_title('Second type img')
        axes[1].axis('off')

        plt.subplots_adjust(wspace=0.5)

        plt.savefig(os.path.join(output_path, f'batch{batch_num}_images_{i}_slice_{central_slice_idx}.png'))
        plt.close()


def measure_gpu_memory(message=""):
    allocated_memory = torch.cuda.memory_allocated()
    reserved_memory = torch.cuda.memory_reserved()
    if message:
        print(f"{message}:")
    print(f"Allocated Memory: {allocated_memory / 1024**2:.2f} MB")
    print(f"Reserved Memory: {reserved_memory / 1024**2:.2f} MB")