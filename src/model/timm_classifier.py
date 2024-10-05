import torch
import torch.nn.functional as F
import pytorch_lightning as L
from torchmetrics import Accuracy
import timm

class TimmClassifier(L.LightningModule):
    def __init__(
        self,
        model_name: str = 'resnet18',
        num_classes: int = 2,
        pretrained: bool = True,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        scheduler_factor: float = 0.1,
        scheduler_patience: int = 10,
        min_lr: float = 1e-6
    ):
        super().__init__()
        self.save_hyperparameters()

        # Load pre-trained model based on model_name
        self.model = timm.create_model(self.hparams.model_name, pretrained=self.hparams.pretrained, num_classes=self.hparams.num_classes)

        # Multi-class accuracy with num_classes
        self.train_acc = Accuracy(task="multiclass", num_classes=self.hparams.num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=self.hparams.num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=self.hparams.num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = F.softmax(logits, dim=1)
        self.train_acc(preds, y)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("val_loss", loss, prog_bar=True)
        self.val_acc(logits, y)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = F.softmax(logits, dim=1)
        self.test_acc(preds, y)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=self.hparams.scheduler_factor,
            patience=self.hparams.scheduler_patience,
            min_lr=self.hparams.min_lr
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }