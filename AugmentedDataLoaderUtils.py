import matplotlib.pyplot as plt
from monai.visualize import matshow3d
import numpy as np

def save_subplot(image, path) -> None:
    first_channel = image[0]
    central_slice_idx = first_channel.shape[1] // 2  # Estraggo l'indice della fetta centrale sul primo canale
                        
    # Seleziona le fette centrali per ciascuna dimensione scalandole tra 0 ed 1
    slice_x = image[:, :, central_slice_idx]
    slice_y = image[:, central_slice_idx, :]
    slice_z = image[central_slice_idx, :, :]

    # Visualizzazione del volume 3D dell'immagine utilizzando matshow3d di MONAI
    fig = plt.figure()
    matshow3d(
        volume=[slice_z, slice_y, slice_x],
        fig=None,
        title="Augmented image",
        figsize=(10, 10),
        # every_n=10,
        frame_dim=-1,
        show=False,
        cmap="gray",
    )
    plt.savefig(path)
    return None