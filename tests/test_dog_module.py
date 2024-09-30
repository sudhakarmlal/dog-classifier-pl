import pytest
from src.datamodules.dog_datamodule import DataModule


def test_catdog_datamodule():
    datamodule = DataModule()

    # Test prepare_data
    #datamodule.prepare_data()

    # Test setup
    datamodule.setup(stage='test')

    # Test dataloaders
    train_loader = datamodule.train_dataloader()
    #val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()

    # Add assertions to check if the dataloaders are correctly set up
    assert len(train_loader) > 0
    #assert len(val_loader) > 0
    assert len(test_loader) > 0