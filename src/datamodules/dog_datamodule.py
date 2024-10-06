import torch
import pytorch_lightning as L
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import gdown
import zipfile
from pathlib import Path
from typing import List

class DataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        num_workers: int = 2,
        batch_size: int = 32,
        splits: List[float] = [0.8, 0.1, 0.1],
        pin_memory: bool = True,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.splits = splits
        self.pin_memory = pin_memory
        
        # Add this block to define the transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def setup(self, stage=None):
        if stage == 'fit' or stage == 'test' or stage == None:
            self._download_and_extract_dataset()
            full_dataset = datasets.ImageFolder(root=self.data_dir + "/dogs_filtered", transform=self.transform)
            n_data = len(full_dataset)
            n_train = int(self.splits[0] * n_data)
            n_val = int(self.splits[1] * n_data)
            n_test = n_data - n_train - n_val
            
            self.train_dataset, self.val_dataset, self.test_dataset = torch.utils.data.random_split(
                full_dataset, [n_train, n_val, n_test]
            )
            
            print(f"Total data: {n_data}")
            print(f"Train data: {len(self.train_dataset)}")
            print(f"Validation data: {len(self.val_dataset)}")
            print(f"Test data: {len(self.test_dataset)}")

    def _download_and_extract_dataset(self):
        file_id = "1-lbK2hPkOeZB-RfwLmeBj3oOSbsWU-ov"
        output_file = "dataset.zip"
        gdown.download(f"https://drive.google.com/uc?id={file_id}", output_file)
        dataset_path = self.data_dir + "/dogs_filtered"
        with zipfile.ZipFile(output_file, 'r') as zip_ref:
            zip_ref.extractall(dataset_path)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )