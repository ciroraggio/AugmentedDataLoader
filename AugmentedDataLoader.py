import torch
import os
import random
from monai.data import ImageDataset
from AugmentedDataLoaderUtils import save_subplot
"""
Inizializza il dataset e genera i batch utilizzando il metodo 'generate_batches'. 
Dopo aver creato un'istanza di AugmentedDataLoader con i parametri necessari, utilizzare il metodo generate_batches() per iterare sui batch di dati.
    - dataset -> Dataset di tipo ImageDataset, contiene: immagini, etichette e trasformazioni sistematiche
    - augmentation_transforms -> Lista di (M) trasformazioni MONAI per l'augmentation
    - batch_size -> Dimensione del batch (K) ovvero i blocchi da restituire
    - num_imgs -> Numero totale di immagini (N)
    - subset_len -> Lunghezza del subset (J)
    - [optional] transformation_device ->  se indicato, è il dispositivo su cui l'utente sceglie di indirizzare la trasformazione
    - [optional] return_device -> se indicato, è il dispositivo su cui l'utente sceglie di indirizzare ogni batch restituito
    - [optional] debug_path -> se indicato, è il path dove l'utente sceglie di salvare una fetta dell'immagine, per ogni immagine dei batch restituiti 
"""

class AugmentedDataLoader:
    def __init__(
        self,
        dataset: ImageDataset,
        augmentation_transforms: list,
        batch_size: int,
        subset_len: int,
        transformation_device: str = "cpu",
        return_device: str = "cpu",
        debug_path: str = None,
    ):
        self.dataset = dataset  # Dataset di tipo ImageDataset, contiene: immagini, segmentazioni o etichette, trasformazioni sistematiche
        self.augmentation_transforms = augmentation_transforms  # Lista di (M) trasformazioni MONAI per l'augmentation
        self.batch_size = batch_size  # Dimensione del batch (K)
        self.num_imgs = len(dataset.image_files)  # Numero totale di immagini (N)
        self.subset_len = subset_len  # Lunghezza del subset (J)
        self.debug_path = debug_path  # se indicato, è il path dove l'utente sceglie di salvare una fetta dell'immagine, per ogni immagine dei batch restituiti
        self.transformation_device = transformation_device # se indicato, è il dispositivo su cui l'utente sceglie di indirizzare la trasformazione
        self.return_device = return_device # se indicato, è il dispositivo su cui l'utente sceglie di indirizzare ogni batch restituito
        self.has_segmentations = bool(dataset.seg_files) # per non trasformare le etichette
        
        if self.dataset is None:
            raise Exception("Dataset is None")
        
        if self.batch_size is None or self.batch_size == 0:
            raise Exception("Invalid batch size")
        
        if self.subset_len is None or self.batch_size == 0:
            raise Exception("Invalid subset len")
    
        
        
    def __iter__(self):
        # Creo una lista contenente tutti gli indici delle immagini ed applico lo shuffle per non operare sulle immagini nello stesso ordine di arrivo
        shuffle_imgs_indices = list(range(self.num_imgs))
        random.shuffle(shuffle_imgs_indices)

        index = 0
        while index < self.num_imgs:
            """
            checked_subset_len determina la lunghezza del subset che sarà estratto,
            se il valore specificato dall'utente per subset_len è maggiore del numero di immagini rimanenti,
            allora subset_len sarà pari al numero di immagini rimanenti, in modo da evitare di superare la lunghezza della lista delle immagini
            """
            remaining_imgs = self.num_imgs - index
            checked_subset_len = min(self.subset_len, remaining_imgs)

            """
            Viene creato un subset di immagini (che vengono scelti dalla lista di indici precedentemente mischiati) avente lunghezza J
            """
            subset_indices = shuffle_imgs_indices[index : index + checked_subset_len]
            subset = torch.utils.data.Subset(self.dataset, subset_indices)
            
            augmented_subset = []
            for data in subset:
                augmented_data = []
                for transformation in self.augmentation_transforms:
                    if(self.transformation_device != "cpu"):
                        torch.cuda.set_device(self.transformation_device)

                    augmented_image = transformation(data[0])
                    # trasformo solo se non è etichetta
                    augmented_segmentation = transformation(data[1]) if self.has_segmentations else data[1]
                    
                    augmented_data.append((augmented_image, augmented_segmentation))

                augmented_subset.append((data[0], data[1]))  # Aggiungo le immagini NON aumentate
                augmented_subset.extend(augmented_data)  # Aggiungo le immagini aumentate

            """
            Ulteriore shuffle delle (J*M)+J immagini, in questo modo riceverà immagini aumentate e non aumentate mixate
            """
            full_batch_indices = torch.randperm(len(augmented_subset))
            augmented_subset = [(augmented_subset[idx][0], augmented_subset[idx][1]) for idx in full_batch_indices]

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
                    for _, data in enumerate(block):
                        image = data[0] 
                        debug_image_path = os.path.join(self.debug_path, f"augmented_image_{image_count}.png")
                        save_subplot(image=image, path=debug_image_path, image_count=image_count)
                        
                        image_count += 1

                float_block = []        
                for tensor_tuple in block:
                    float_img = tensor_tuple[0].float()
                    float_seg = tensor_tuple[1].float() if self.has_segmentations else tensor_tuple[1]
                    float_block.append([float_img,float_seg])  # Conversione dei tensori delle immagini e segmentazioni a float32

                images = torch.stack([data[0] for data in float_block]).to(self.return_device)
                segmentations_or_labels = torch.stack([data[1] for data in float_block]).to(self.return_device) if self.has_segmentations else [data[1] for data in float_block]
                
                yield images, segmentations_or_labels

            """
            Dopo aver passato tutte le (J*M)+J immagini a blocchi di K, incremento l'indice e riparto per leggere altre J immagini
            """
            index += checked_subset_len