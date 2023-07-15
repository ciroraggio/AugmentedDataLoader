import torch
import os
import random
from monai.data import ImageDataset
import torchvision.utils as vutils

"""
Inizializza il dataset e genera i batch utilizzando il metodo 'generate_batches'. 
Dopo aver creato un'istanza di AugmentedDataLoader con i parametri necessari, utilizzare il metodo generate_batches() per iterare sui batch di dati.
    - dataset -> Dataset di tipo ImageDataset, contiene: immagini, etichette e trasformazioni sistematiche
    - augmentation_transforms -> Lista di (M) trasformazioni MONAI per l'augmentation
    - batch_size -> Dimensione del batch (K) ovvero i blocchi da restituire
    - num_patients -> Numero totale di pazienti (N)
    - subset_len -> Lunghezza del subset (J)
    - [optional] debug_path -> se indicato, è il path dove l'utente sceglie di salvare una fetta dell'immagine, per ogni immagine dei batch restituiti 
"""


class AugmentedDataLoader:
    def __init__(
        self,
        dataset: ImageDataset,
        augmentation_transforms: list,
        batch_size: int,
        subset_len: int,
        debug_path: str = None,
    ):
        self.dataset = dataset  # Dataset di tipo ImageDataset, contiene: immagini, etichette e trasformazioni sistematiche
        self.augmentation_transforms = augmentation_transforms  # Lista di (M) trasformazioni MONAI per l'augmentation
        self.batch_size = batch_size  # Dimensione del batch (K)
        self.num_patients = len(dataset.image_files)  # Numero totale di pazienti (N)
        self.subset_len = subset_len  # Lunghezza del subset (J)
        self.debug_path = debug_path  # se indicato, è il path dove l'utente sceglie di salvare una fetta dell'immagine, per ogni immagine dei batch restituiti

    def generate_batches(self):
        if self.dataset is None:
            raise Exception("Dataset is None")

        if (
            self.dataset.image_files
            and self.dataset.seg_files
            and len(self.dataset.image_files) != len(self.dataset.seg_files)
        ):
            raise Exception("The length of the images and segmentations don't match")

        # Creo una lista contenente tutti gli indici dei pazienti ed applico lo shuffle per non operare sui pazienti nello stesso ordine di arrivo
        shuffle_patient_indices = list(range(self.num_patients))
        random.shuffle(shuffle_patient_indices)

        index = 0
        while index < self.num_patients:
            """
            ***checked_subset_len*** determina la lunghezza del subset che sarà estratto,
            se il valore specificato dall'utente per subset_len è maggiore del numero di pazienti rimanenti,
            allora subset_len sarà pari al numero di pazienti rimanenti, in modo da evitare di superare la lunghezza della lista dei pazienti
            """
            remaining_patients = self.num_patients - index
            checked_subset_len = min(self.subset_len, remaining_patients)

            """
            Viene creato un subset di pazienti (che vengono scelti dalla lista di indici precedentemente mischiati) avente lunghezza J
            """
            subset_indices = shuffle_patient_indices[index : index + checked_subset_len]
            subset = torch.utils.data.Subset(self.dataset, subset_indices)

            augmented_subset = []
            for data in subset:
                augmented_data = []
                for transformation in self.augmentation_transforms:
                    augmented_data.append(transformation(data[0]))

                augmented_subset.append(data[0])  # Aggiungo le immagini NON aumentate
                augmented_subset.extend(
                    augmented_data
                )  # Aggiungo le immagini aumentate

            """
            Ulteriore shuffle delle (J*M)+J immagini, in questo modo riceverà immagini aumentate e non aumentate mixate
            """
            full_batch_indices = torch.randperm(len(augmented_subset))
            augmented_subset = [augmented_subset[idx] for idx in full_batch_indices]

            """
            Creo blocchi di dimensione K
            - range(0, len(augmented_subset), self.batch_size): 
                genera una sequenza di valori che rappresentano gli indici di inizio di ogni blocco. 
                Gli indici partono da 0 e avanzano con un passo pari a self.batch_size, fino a raggiungere la lunghezza totale di augmented_subset.
            - augmented_subset[i:i + self.batch_size]: 
                seleziona una sotto-lista di augmented_subset che va dall'indice i fino all'indice i + self.batch_size. 
                Questo crea un blocco di immagini di dimensione self.batch_size.
            """
            blocks = [
                augmented_subset[i : i + self.batch_size]
                for i in range(0, len(augmented_subset), self.batch_size)
            ]
            
            image_count = 0
            for block in blocks:
                """
                Restituisco i blocchi di dimensione K.
                Per ogni blocco, viene creato un tensore batch utilizzando la funzione stack di torch che concatena i tensori all'interno del blocco lungo la dimensione 0,
                creando così un unico tensore che rappresenta un batch di immagini.
                Operatore yield per mantenere lo stato della funzione tra le chiamate.
                """
                if self.debug_path:
                    for i, data in enumerate(block):
                        image = data[0]
                        central_slice = image[image.shape[0] // 2]  # Estraggo la fetta centrale sul primo canale
                        normalized_slice = (central_slice - central_slice.min()) / (central_slice.max() - central_slice.min())
                        debug_image_path = os.path.join(self.debug_path, f"augmented_image_{image_count}.png")
                        vutils.save_image(normalized_slice, debug_image_path)
                        image_count += 1

                        
                block = [data.float() for data in block]  # Conversione dei tensori delle immagini a float32
                batch = torch.stack(block)
                yield batch

            """
            Dopo aver passato tutti i (J*M)+J pazienti a blocchi di K, incremento l'indice e riparto per leggere altri J pazienti
            """
            index += checked_subset_len
