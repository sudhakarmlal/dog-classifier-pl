import rootutils

# Setup root directory
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import hydra
from omegaconf import DictConfig
import torch
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from rich.progress import track
from hydra.utils import instantiate
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import Logger
from pytorch_lightning.callbacks import Callback

from src.utils.task_wrapper import task_wrapper
from src.model.timm_classifier import TimmClassifier  # Added import for TimmClassifier

@task_wrapper
def infer(cfg: DictConfig) -> None:
    # Setup logger
    logger: Logger = instantiate(cfg.logger)

    # Setup callbacks
    callbacks: list[Callback] = instantiate(cfg.callbacks)

    # Load the model
    #model = TimmClassifier.load_from_checkpoint(cfg.ckpt_path+"/"+"model_hy.ckpt")

    model=TimmClassifier.load_from_checkpoint('checkpoint/model_hy.ckpt')
    model.eval()

    # Prepare transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Create output folder if it doesn't exist
    output_path = Path(cfg.output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get all image files
    input_path = Path(cfg.input_folder)
    image_files = list(input_path.glob('*.[pj][np][g]'))

    # Class labels
    class_labels = ['Beagle', 'Bulldog', 'German_Shepherd', 'Labrador_Retriever', 'Rottweiler', 'Boxer', 'Dachshund', 'Golden_Retriever', 'Poodle', 'Yorkshire_Terrier']

    # Process each image
    for image_file in track(image_files, description="Processing images"):
        # Load and preprocess the image
        img = Image.open(image_file).convert("RGB")
        img_tensor = transform(img).unsqueeze(0)

        # Perform inference
        with torch.no_grad():
            output = model(img_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()

        # Get predicted label
        predicted_label = class_labels[predicted_class]

        # Create and save the plot
        plt.figure(figsize=(8, 8))
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"Predicted: {predicted_label} (Confidence: {confidence:.2f})")
        output_file = output_path / f"{image_file.stem}_prediction.png"
        plt.savefig(output_file)
        plt.close()

        # Log the prediction
        if hasattr(logger, 'log_metrics'):
            logger.log_metrics({
                "predicted_class": predicted_class,
                "confidence": confidence
            })
        else:
            print("Warning: logger.log_metrics not available. Skipping metric logging.")
            # Optionally, use a different logging method or just print the metrics

    # Finalize the logger
    if hasattr(logger, 'finalize'):
        logger.finalize("success")
    else:
        # Use a standard logging method instead
        import logging
        logging.info("Inference completed successfully")

@hydra.main(version_base=None, config_path="../config", config_name="infer")
def main(cfg: DictConfig):
    print(f"Checkpoint path: {cfg.ckpt_path}")
    infer(cfg)

if __name__ == "__main__":
    main()