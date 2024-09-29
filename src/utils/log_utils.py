import os
import sys
from functools import wraps
import torch
from torchviz import make_dot
from PIL import Image
from loguru import logger
from rich.progress import Progress, SpinnerColumn, TextColumn

def visualize_model(model):
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    dot = make_dot(y, params=dict(model.named_parameters()))
    dot.render("model_architecture", format="png")
    Image.open("model_architecture.png").show()

def setup_logging():
    log_folder = "logs"
    os.makedirs(log_folder, exist_ok=True)
    log_file = os.path.join(log_folder, "dog_classifier.log")
    
    logger.remove()  # Remove default handler
    logger.add(sys.stderr, format="{time} {level} {message}", level="INFO")
    logger.add(log_file, rotation="10 MB", level="DEBUG")
    
    return logger

def task_wrapper(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = setup_logging()
        try:
            logger.info(f"Starting task: {func.__name__}")
            result = func(*args, **kwargs)
            logger.info(f"Task completed: {func.__name__}")
            return result
        except Exception as e:
            logger.exception(f"Error in task {func.__name__}: {str(e)}")
            raise
    return wrapper

def get_rich_progress():
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True
    )




