# scripts/train_multimodal.py
import pandas as pd
import torch
from torch import nn, optim
from transformers import AutoModel, AutoTokenizer
from torchvision import models, transforms
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

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

    def forward(self, text_embeds, image_embeds, tabular_data):
        text_features = self.text_fc(text_embeds)
        image_features = self.image_fc(image_embeds)
        combined = torch.cat((text_features, image_features, tabular_data), dim=1)
        return self.fc(combined)

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
        model.train()
        total_loss = 0
        for i, row in X_train.iterrows():
            # Load and process text data
            text_embed = encode_text(row['description'], tokenizer, text_model)

            # Load and process image data
            image_path = f"/home/mohsn/ml_optimization_multimodal/data/spacecraft_images/{row['description'].replace(' ', '_').lower()}.jpg"
            image_embed = load_image(image_path)

            # Convert tabular data
            tabular_data = torch.tensor([row['source_id'], row['quantity']], dtype=torch.float).unsqueeze(0)

            # Forward pass
            output = model(text_embed, image_embed, tabular_data)
            target = torch.tensor([row['target']], dtype=torch.float)

            # Calculate loss and optimize
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(X_train)}")

    # Validation
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for _, row in X_val.iterrows():
            text_embed = encode_text(row['description'], tokenizer, text_model)
            image_path = f"/home/mohsn/ml_optimization_multimodal/data/spacecraft_images/{row['description'].replace(' ', '_').lower()}.jpg"
            image_embed = load_image(image_path)
            tabular_data = torch.tensor([row['source_id'], row['quantity']], dtype=torch.float).unsqueeze(0)

            output = model(text_embed, image_embed, tabular_data)
            preds.append(output.item())
            targets.append(row['target'])

    # Calculate Mean Absolute Error
    mae = mean_absolute_error(targets, preds)
    print(f"Validation MAE: {mae}")

if __name__ == "__main__":
    train_multimodal_model()
