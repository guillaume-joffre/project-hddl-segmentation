import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(r"""
    # Mini-projet 3 : ViT vs CNN
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Import des packages
    """)
    return


@app.cell
def _():
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    import torchvision
    import torchvision.transforms as transforms
    import torchvision.datasets as datasets

    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    from tqdm import tqdm
    import time
    import random
    import os

    from datasets import load_dataset
    return (
        DataLoader,
        Dataset,
        F,
        load_dataset,
        nn,
        optim,
        torch,
        tqdm,
        transforms,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## Choix des datasets et justifications
    """)
    return


@app.cell
def _(load_dataset):
    food_dataset = load_dataset("Kaludi/food-category-classification-v2.0")

    print(food_dataset)
    print(food_dataset["train"][0])
    return (food_dataset,)


@app.cell
def _(DataLoader, Dataset, food_dataset, transforms):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # ou la taille attendue par ton modèle
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Dataset PyTorch pour Hugging Face
    class FoodDataset(Dataset):
        def __init__(self, hf_dataset, transform=None):
            self.dataset = hf_dataset
            self.transform = transform

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            img = self.dataset[idx]['image']
            label = self.dataset[idx]['label']
            img = img.convert('RGB')  # Ajoute cette ligne pour forcer le RGB
            if self.transform:
                img = self.transform(img)
            return img, label

    # Instancie les datasets et DataLoaders
    train_dataset = FoodDataset(food_dataset['train'], transform=transform)
    val_dataset = FoodDataset(food_dataset['validation'], transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    return train_loader, val_loader


@app.cell
def _(mo):
    mo.md(r"""
    ## Explication de l'architecture Vision Transformers
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Implémentation d'une architecure ViT
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Implémentation d'une architecture CNN
    """)
    return


@app.cell
def _(F, nn):
    class SimpleCNN(nn.Module):
        def __init__(self, num_classes=10):
            super(SimpleCNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(64 * 16 * 16, 128)
            self.fc2 = nn.Linear(128, num_classes)
            self.dropout = nn.Dropout(0.5)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x
    return (SimpleCNN,)


@app.cell
def _(SimpleCNN, food_dataset, nn, optim, torch):
    # Détection automatique du nombre de classes à partir du dataset
    num_classes = len(set([sample['label'] for sample in food_dataset['train']]))

    # Gestion du device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Utilisation du device : {device}")

    # Instanciation du modèle CNN
    model_cnn = SimpleCNN(num_classes=num_classes).to(device)

    # Définition de la fonction de perte et de l'optimiseur
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_cnn.parameters(), lr=1e-3)
    return criterion, device, model_cnn, optimizer


@app.cell
def _(
    criterion,
    device,
    model_cnn,
    optimizer,
    torch,
    tqdm,
    train_loader,
    val_loader,
):
    # Boucle d'entraînement et de validation pour le CNN
    def train_one_epoch(model, loader, criterion, optimizer, device):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in tqdm(loader, desc='Train', leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    def evaluate(model, loader, criterion, device):
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in tqdm(loader, desc='Val', leave=False):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    # Entraînement du modèle
    num_epochs = 10
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model_cnn, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model_cnn, val_loader, criterion, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Comparaison des modèles selon les datasets
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Conclusion
    """)
    return


if __name__ == "__main__":
    app.run()
