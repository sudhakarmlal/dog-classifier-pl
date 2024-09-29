import argparse
import torch
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from model.dog_classifier import DogClassifier
from utils.log_utils import setup_logging
from utils.task_wrapper import task_wrapper
from rich.progress import track

@task_wrapper
def infer(input_folder: str, output_folder: str, ckpt_path: str):
    # Set up logging
    log_dir = Path("logs")
    setup_logging()

    # Load the model
    model = DogClassifier.load_from_checkpoint(ckpt_path)
    model.eval()

    # Prepare transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Create output folder if it doesn't exist
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get all image files
    input_path = Path(input_folder)
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Infer dog breeds from images")
    parser.add_argument("--input_folder", type=str, required=True, help="Path to the folder containing input images")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to the folder to save predictions")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to the model checkpoint")
    args = parser.parse_args()

    infer(args.input_folder, args.output_folder, args.ckpt_path)