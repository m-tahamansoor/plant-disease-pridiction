# src/train.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from tqdm import tqdm

# Import modules from the local 'src' package
from src.config import EPOCHS, CHECKPOINT_PATH, LABELS_PATH, DEVICE
from src.data_loader import create_dataloaders
from src.model import build_model

def train_epoch(model, dataloader, criterion, optimizer):
    """Handles the training loop for one epoch."""
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for inputs, labels in tqdm(dataloader, desc="Training"):
        # Move data to the configured device (GPU)
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct_predictions += torch.sum(preds == labels.data).item()
        total_samples += inputs.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples
    return epoch_loss, epoch_acc

def validate_epoch(model, dataloader, criterion):
    """Handles the validation loop for one epoch."""
    model.eval() # Set model to evaluation mode
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    # Disable gradient calculations during validation
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validation"):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_predictions += torch.sum(preds == labels.data).item()
            total_samples += inputs.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples
    return epoch_loss, epoch_acc


def main():
    """
    Main function to load data, build, train, and save the PyTorch model.
    """
    print(f"🚀 Starting PyTorch Training on Device: {DEVICE}")
    
    # --- 1. Data Loading ---
    train_dataloader, val_dataloader, class_names = create_dataloaders()
    num_classes = len(class_names)
    
    # --- 2. Model Building ---
    model = build_model(num_classes, DEVICE)

    # --- 3. Loss, Optimizer, and Tracking ---
    # PyTorch uses CrossEntropyLoss which combines Softmax and NLLLoss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_val_loss = float('inf')
    
    # Storage for history (for later plotting)
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    # --- 4. Training Loop ---
    print("\nStarting Training...")
    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
        
        train_loss, train_acc = train_epoch(model, train_dataloader, criterion, optimizer)
        val_loss, val_acc = validate_epoch(model, val_dataloader, criterion)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Validation Loss: {val_loss:.4f}, Validation Acc: {val_acc:.4f}")

        # Store history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        # Save the best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save the model state dictionary (standard PyTorch practice)
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"Model saved to {CHECKPOINT_PATH} (Val Loss: {val_loss:.4f})")
    
    # --- 5. Save Labels ---
    with open(LABELS_PATH, 'w') as f:
        for class_name in class_names:
            f.write(class_name + '\n')

    print(f"\n✅ Training completed!")
    print(f"✅ Best model state dictionary saved at: {CHECKPOINT_PATH}")
    print(f"✅ Labels saved at: {LABELS_PATH}")
    
    # Note: Returning history and model for potential visualization
    return history, model 

if __name__ == '__main__':
    main()