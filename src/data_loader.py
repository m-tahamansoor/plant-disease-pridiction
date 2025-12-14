# src/data_loader.py

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src.config import TRAIN_PATH, VALID_PATH, IMG_SIZE, BATCH_SIZE
import os

def create_dataloaders():

    # 1. Define Transformations 
    train_transforms = transforms.Compose([
        # IMG_SIZE is now 300
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),              
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])

    val_transforms = transforms.Compose([
        # IMG_SIZE is now 300
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 2. Create Datasets using ImageFolder
    print("Loading Training Data...")
    train_dataset = datasets.ImageFolder(str(TRAIN_PATH), transform=train_transforms)
    
    print("\nLoading Validation Data...")
    val_dataset = datasets.ImageFolder(str(VALID_PATH), transform=val_transforms)

    # Get Class Names
    class_names = train_dataset.classes
    print(f"\n✅ Detected {len(class_names)} Classes: {class_names[:5]}...")

    # 3. Create DataLoaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=os.cpu_count(), # Use all CPU cores for efficient data loading
        pin_memory=True # Helps speed up transfer to CUDA
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=os.cpu_count(),
        pin_memory=True
    )
    
    return train_dataloader, val_dataloader, class_names