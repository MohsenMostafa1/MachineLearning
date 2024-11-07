# MachineLearning

 The script begins by importing necessary libraries: 
 
 ```python
import pandas as pd
import torch
from torch import nn, optim
from transformers import AutoModel, AutoTokenizer
from torchvision import models, transforms
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
```


The text model uses BERT to extract features from text input.
The image model uses ResNet-18 to extract features from images.
A fully connected layer combines features from both models along with two additional inputs from tabular data.
    
 ```python
class MultimodalModel(nn.Module):
    def __init__(self):
        super(MultimodalModel, self).__init__()
        # Text model (using transformers)
        self.text_model = AutoModel.from_pretrained("bert-base-uncased")
        self.text_fc = nn.Linear(768, 256)

        # Image model (using ResNet)
        resnet = models.resnet18(pretrained=True)
        self.image_model = nn.Sequential(*list(resnet.children())[:-1])
        self.image_fc = nn.Linear(resnet.fc.in_features, 256)

        # Final regression layer
        self.fc = nn.Linear(256 * 2 + 2, 1)
```


load_image: Loads an image from disk, resizes it to 224x224 pixels (the input size for ResNet), and converts it to a tensor.
encode_text: Tokenizes the input text description and extracts the BERT embeddings.
    
 ```python
def load_image(image_path):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert("RGB")
    return preprocess(image).unsqueeze(0)

def encode_text(description, tokenizer, model):
    inputs = tokenizer(description, return_tensors="pt")
    return model(**inputs).last_hidden_state[:, 0, :]
```


The function reads a CSV file containing the dataset and splits it into training and validation sets.
It initializes the tokenizer and text model from Hugging Face's library.
The optimizer (Adam) and loss function (Mean Squared Error) are set up.
    
 ```python
def train_multimodal_model():
    # Load and split data
    data = pd.read_csv('/home/mohsn/ml_optimization_multimodal/data/candidates_data.csv')
    X_train, X_val = train_test_split(data, test_size=0.2, random_state=42)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    text_model = AutoModel.from_pretrained("bert-base-uncased")
    model = MultimodalModel()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()  # Mean Squared Error as the loss function

    # Training loop
    for epoch in range(10):  # Adjust epochs as needed
        ...
```
