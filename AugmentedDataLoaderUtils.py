import matplotlib.pyplot as plt
from monai.visualize import matshow3d

def save_subplot(image, path, image_count) -> None:
    first_channel = image[0]
    central_slice_idx = first_channel.shape[1] // 2  # Extract the central slice index on the first channel
                        
    # Select the central slices for each dimension
    slice_x = image[:, :, central_slice_idx]
    slice_y = image[:, central_slice_idx, :]
    slice_z = image[central_slice_idx, :, :]

    # 3D volume visualization of image using MONAI's matshow3d    
    fig = plt.figure()
    matshow3d(
        volume=[slice_z, slice_y, slice_x],
        fig=None,
        title=f"Augmented image n. {image_count} for the current block",
        figsize=(10, 10),
        # every_n=10,
        frame_dim=-1,
        show=False,
        cmap="gray",
    )
    plt.savefig(path)
    return None