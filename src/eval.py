import os
import lightning.pytorch as pl
from datamodules.dog_datamodule import DataModule
from model.dog_classifier import DogClassifier

def main():
    # Setup data module
    datamodule = DataModule()
    datamodule.setup(stage='test')

    # Load the best model
    checkpoint_path = '/opt/logs/checkpoint/model_tr.ckpt'
    model = DogClassifier.load_from_checkpoint(checkpoint_path)

    # Initialize trainer
    trainer = pl.Trainer(accelerator="auto")

    # Evaluate the model
    trainer.test(model, datamodule=datamodule)

if __name__ == '__main__':
    main()



