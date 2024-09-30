import torch
import lightning.pytorch as pl
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import gdown
import zipfile

class DataModule(pl.LightningDataModule):
    def __init__(self, transform=None, batch_size=32):
        super().__init__()
        self.root_dir = "dataset"
        self.transform = transform or transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(),
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.batch_size = batch_size

    def setup(self, stage=None):
        if stage == 'fit' or stage == 'test':
            if stage == 'fit':
                self._download_and_extract_dataset()
            dataset = datasets.ImageFolder(root=self.root_dir, transform=self.transform)
            n_data = len(dataset)
            print(n_data)
            n_train = int(0.8 * n_data)
            n_test = n_data - n_train
            print("Train data lenght",n_train)
            print("Test data length",n_test)
            train_dataset, test_dataset = torch.utils.data.random_split(dataset, [n_train, n_test])
            self.train_dataset = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            self.test_dataset = DataLoader(test_dataset, batch_size=self.batch_size)

    def _download_and_extract_dataset(self):
        file_id = "1-lbK2hPkOeZB-RfwLmeBj3oOSbsWU-ov"
        output_file = "dataset.zip"
        gdown.download(f"https://drive.google.com/uc?id={file_id}", output_file)
        
        with zipfile.ZipFile(output_file, 'r') as zip_ref:
            zip_ref.extractall(self.root_dir)

    def train_dataloader(self):
        #return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        return self.train_dataset

    def test_dataloader(self):
        #return DataLoader(self.test_dataset, batch_size=self.batch_size)
        return self.test_dataset