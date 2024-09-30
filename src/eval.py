import lightning.pytorch as pl
import torch
from sklearn.metrics import classification_report
import numpy as np
from datamodules.dog_datamodule import DataModule
from model.dog_classifier import DogClassifier
from utils.log_utils import setup_logging
from utils.task_wrapper import task_wrapper

logger = setup_logging()



@task_wrapper
def eval(ckpt_path):
    print("ABC")
    class_names=['Beagle','Bulldog','German_Shepherd','Labrador_Retriever', 'Rottweiler','Boxer','Dachshund','Golden_Retriever','Poodle','Yorkshire_Terrier']
    device = torch.device("cpu")   #"cuda:0"
    #data_module = DogDataModule()
    #model = DogClassifier(lr=1e-3)
    model = DogClassifier().to(device)
    # create model and load state dict
    #model.load_state_dict(torch.load("logs/model_tr.ckpt"))
    model.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cpu'))['state_dict'])
    model.eval()
    y_true=[]
    y_pred=[]
    datamodule = DataModule()
    datamodule.setup(stage='test')
    with torch.no_grad():
        for test_data in datamodule.test_dataloader():
            test_images, test_labels = test_data[0].to(device), test_data[1].to(device)
            pred = model(test_images).argmax(dim=1)
            for i in range(len(pred)):
                y_true.append(test_labels[i].item())
                y_pred.append(pred[i].item())
    print("My report")   
    labels=[1,2,3,4,5,6,7,8,9,10]         
    print(classification_report(y_true,y_pred,target_names=class_names,labels=labels,zero_division=1.0,digits=4))

if __name__ == "__main__":
    ckpt_path = "/opt/logs/checkpoint/model_pr.ckpt"  # Update this path
    eval(ckpt_path)