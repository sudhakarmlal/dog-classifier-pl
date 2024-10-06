import os
import rootutils

# Setup the root directory
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
import torch


#from sklearn.metrics import classification_report
import numpy as np
from src.utils.log_utils import setup_logging
from src.utils.task_wrapper import task_wrapper

logger = setup_logging()

@task_wrapper
@hydra.main(version_base="1.3", config_path="../config", config_name="eval")
def eval(cfg: DictConfig) -> None:
    # Setup paths
    print(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule = hydra.utils.instantiate(cfg.data)

    print(f"Instantiating model <{cfg.model._target_}>")
    model = hydra.utils.instantiate(cfg.model)

    # Setup callbacks
    callbacks = []
    if "callbacks" in cfg:
        for _, cb_conf in cfg.callbacks.items():
            if "_target_" in cb_conf:
                print(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Setup logger
    logger = []
    if "logger" in cfg:
        for _, lg_conf in cfg.logger.items():
            if "_target_" in lg_conf:
                print(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))

    # Load model from checkpoint
    ckpt_path = "checkpoint/model_hy.ckpt"
    model.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cpu'))['state_dict'])
    #model.eval()

    # Setup trainer
    #trainer_cfg = OmegaConf.to_container(cfg.trainer, resolve=True)
    #trainer_cfg.pop('_target_', None)

    #trainer = pl.Trainer(**trainer_cfg)
    print(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, logger=logger, _convert_="partial"
    )


    #best_model_path = trainer.checkpoint_callback.best_model_path if hasattr(trainer, 'checkpoint_callback') else None
    #model.load_state_dict(torch.load(best_model_path, map_location=torch.device('cpu'))['state_dict'])
    #model.eval()
    # Evaluate the model
    trainer.test(model=model, datamodule=datamodule)

    # Additional custom evaluation
    y_true = []
    y_pred = []
    device = torch.device("cpu")
    
    with torch.no_grad():
        for test_data in datamodule.test_dataloader():
            test_images, test_labels = test_data[0].to(device), test_data[1].to(device)
            pred = model(test_images).argmax(dim=1)
            for i in range(len(pred)):
                y_true.append(test_labels[i].item())
                y_pred.append(pred[i].item())

    #print("Classification Report:")
    #class_names = datamodule.class_names  # Assuming your DataModule has a class_names attribute
    #labels = list(range(1, len(class_names) + 1))
    #print(classification_report(y_true, y_pred, target_names=class_names, labels=labels, zero_division=1.0, digits=4))

if __name__ == "__main__":
    eval()