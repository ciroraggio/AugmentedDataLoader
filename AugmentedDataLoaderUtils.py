import matplotlib.pyplot as plt
import numpy as np

def save_subplot(images, central_slice_idx, path):
    fig, axs = plt.subplots(1, 3)
    print(np.min(images[0]), np.max(images[0]))
    axs[0].imshow(images[0][:, :, central_slice_idx], cmap='gray', vmin=0, vmax=1)
    axs[0].axis('off')

    axs[1].imshow(images[1][:, :, central_slice_idx], cmap='gray', vmin=0, vmax=1)
    axs[1].axis('off')

    axs[2].imshow(images[2][:, :, central_slice_idx], cmap='gray', vmin=0, vmax=1)
    axs[2].axis('off')

    plt.savefig(path)
    plt.close()


def zero_one_scaling(slice):
    return (slice - slice.min()) / (slice.max() - slice.min())