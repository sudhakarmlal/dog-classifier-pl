import os
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar, RichModelSummary
from lightning.pytorch.loggers import TensorBoardLogger
from datamodules.dog_datamodule import DataModule
from model.dog_classifier import DogClassifier
from utils.log_utils import setup_logging
from utils.task_wrapper import task_wrapper

@task_wrapper
def train_and_test():
    logger = setup_logging()
    logger.info("Starting training process")

    # Setup data module
    datamodule = DataModule()
    
    # Initialize model
    model = DogClassifier()

    # Setup checkpoint callback
    #checkpoint_callback = ModelCheckpoint(
      #  dirpath='logs/checkpoints',
       # filename='model_r',
       # save_top_k=3,
       # monitor='val_loss'
    #)

    # Setup logger
    tb_logger = TensorBoardLogger('logs', name='dog_classifier')

    # Initialize trainer
    checkpoint_callback = ModelCheckpoint(save_on_train_epoch_end=True,dirpath='/opt/logs/checkpoint',filename="model_pr")
    trainer = Trainer(
        max_epochs=10,
        callbacks=[
            checkpoint_callback,
            RichProgressBar(),
            RichModelSummary(max_depth=2)
        ],
        logger=tb_logger,
        log_every_n_steps=10,
        accelerator="auto",
    )

    # Train the model
    logger.info("Starting model training")
    trainer.fit(model, datamodule)
    logger.info("Model training completed")

    # Test the model
    logger.info("Starting model testing")
    trainer.test(model, datamodule)
    logger.info("Model testing completed")

if __name__ == '__main__':
    train_and_test()
