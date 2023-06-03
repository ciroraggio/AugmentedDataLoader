import torch
from monai.transforms import Resize
from monai.data import ImageDataset

"""
Inizializza il dataset e genera i batch utilizzando il metodo 'generate_batches'. 
Dopo aver creato un'istanza di DataGenerator con i parametri necessari, utilizzare il metodo generate_batches() per iterare sui batch di dati.
    - images_to_transform -> Lista delle immagini di (N) pazienti
    - augmentation_transforms -> Lista di (M) trasformazioni MONAI per l'augmentation
    - batch_size -> Dimensione del batch (K) ovvero i blocchi da restituire
    - num_patients -> Numero totale di pazienti (N)
"""

class DataGenerator:
    def __init__(self, images_to_transform, augmentation_transforms, batch_size, target_size=(512,512,1)):
        self.images_to_transform = images_to_transform  # Lista delle immagini di (N) pazienti
        self.augmentation_transforms = augmentation_transforms  # Lista di (M) trasformazioni MONAI per l'augmentation
        self.batch_size = batch_size  # Dimensione del batch (K)
        self.num_patients = len(images_to_transform)  # Numero totale di pazienti (N)
        self.target_size = target_size
        self.dataset = None

    def initialize_dataset(self):
        resize_transform = Resize(self.target_size)
        self.dataset = ImageDataset(image_files=self.images_to_transform, transform=resize_transform)

    def generate_batches(self):
        if self.dataset is None:
            self.initialize_dataset()

        """
        Vengono generati indici casuali per selezionare un sottoinsieme di (J) pazienti dal numero totale di pazienti (N). 
        """
        indices = torch.randperm(self.num_patients)
        index = 0

        while index < self.num_patients:
            """
            Nel caso in cui la somma dell'indice corrente (index) e la dimensione del batch (self.batch_size) sia minore del numero totale di pazienti (self.num_patients), 
            allora end_index sarà indice corrente + dimensione del batch.
            Se invece la somma supera il numero totale di pazienti, allora end_index sarà il numero totale di pazienti stesso.
            In questo modo, è sicuro che il subset di pazienti non superi mai il numero totale di pazienti disponibili.
            """
            end_index = min(index + self.batch_size, self.num_patients)

            """
            Viene creato un subset di J pazienti selezionando gli indici generati casualmente.
            """
            subset_indices = indices[index:end_index]
            subset = torch.utils.data.Subset(self.dataset, subset_indices)

            """
            Applico a tutti i J pazienti le M trasformazioni, per avere (J*M)+J immagini
            """
            augmented_subset = []
            for data in subset:
                augmented_data = []
                for transformation in self.augmentation_transforms:
                    augmented_data.append(transformation(data))
                augmented_subset.append(data)  # Aggiungo le immagini NON aumentate
                augmented_subset.extend(augmented_data)  # Aggiungo le immagini aumentate

            # Shuffle delle (J*M)+J immagini
            indices = torch.randperm(len(augmented_subset))
            augmented_subset = [augmented_subset[idx] for idx in indices]
 
            """
            Creo blocchi di dimensione K
            - range(0, len(augmented_subset), self.batch_size): 
                genera una sequenza di valori che rappresentano gli indici di inizio di ogni blocco. 
                Gli indici partono da 0 e avanzano con un passo pari a self.batch_size, fino a raggiungere la lunghezza totale di augmented_subset.
            - augmented_subset[i:i + self.batch_size]: 
                seleziona una sotto-lista di augmented_subset che va dall'indice i fino all'indice i + self.batch_size. 
                Questo crea un blocco di immagini di dimensione self.batch_size.
            """
            blocks = [augmented_subset[i:i + self.batch_size] for i in range(0, len(augmented_subset), self.batch_size)]

            for block in blocks:
                """
                Restituisco i blocchi di dimensione K.
                Per ogni blocco, viene creato un tensore batch utilizzando la funzione stack di torch che concatena i tensori all'interno del blocco lungo la dimensione 0, 
                creando così un unico tensore che rappresenta un batch di immagini.
                Operatore yield per mantenere lo stato della funzione tra le chiamate. 
                """
                batch = torch.stack(block)
                yield batch

            """
            Dopo aver passato tutti i (J*M)+J pazienti a blocchi di K, incremento l'indice e riparto per leggere altri J pazienti
            """
            index += self.batch_size 