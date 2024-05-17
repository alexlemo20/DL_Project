import numpy as np
import matplotlib.pyplot as plt
from helper_methods import get_train_val, create_dataset, show_images

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet18_Weights
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoImageProcessor, ViTForImageClassification
from transformers import ViTImageProcessor
# $env:TF_ENABLE_ONEDNN_OPTS = "0"

#from datasets import load_dataset

#dataset = load_dataset("huggingface/cats-image")
#image = dataset["test"]["image"][0]
#image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
#model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
#inputs = image_processor(image, return_tensors="pt")
#with torch.no_grad():
#    logits = model(**inputs).logits

# model predicts one of the 1000 ImageNet classes
#predicted_label = logits.argmax(-1).item()
#print(model.config.id2label[predicted_label])

if __name__ == "__main__":
    
    df_train, df_val = get_train_val(filepath='C:\\Users\\alexa\\Documents\\Deep Learning\\Project\\datasets\\annotations\\annotations\\trainval.txt', val_size=0.2)
    # C:\\Users\\alexa\\Documents\\Deep Learning\\Project\\datasets\\images\\images\\

    X_train, Y_train = create_dataset(df_train, base_path='C:\\Users\\alexa\\Documents\\Deep Learning\\Project\\datasets\\images\\images\\')
    X_val, Y_val = create_dataset(df_val, base_path='C:\\Users\\alexa\\Documents\\Deep Learning\\Project\\datasets\\images\\images\\')

    Y_train = Y_train.long()
    Y_val = Y_val.long()
    
    #model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
    #with torch.no_grad():
    #    logits = model(**X_train).logits # may need preprocessing of training data
    #predicted_label = logits.argmax(-1).item()
    #print(model.config.id2label[predicted_label])
    

    #print(X_train.shape)
    #img = X_train[0].permute(1, 2, 0).numpy()
    #print(img.shape)
    #print(img)
    #show_images(X_train, Y_train, 1)
    model_name_or_path = 'google/vit-base-patch16-224-in21k'
    #processor = ViTImageProcessor.from_pretrained(model_name_or_path)
    #print(processor)
    #tensor_image = processor(img, return_tensors='pt')
    #print(tensor_image)

    from transformers import ViTForImageClassification
    labels = torch.unique(Y_train)
    labels = labels.tolist()
    model = ViTForImageClassification.from_pretrained(
        model_name_or_path,
        num_labels=torch.max(Y_train) + 1,
        id2label={str(i): c for i, c in enumerate(labels)},
        label2id={c: str(i) for i, c in enumerate(labels)}
    )