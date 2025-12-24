# train_ann.py
import torch
import torch.nn as nn
import torch.optim as optim
from src.config import DEVICE, IMG_SIZE, BATCH_SIZE, EPOCHS
from src.data_loader import create_dataloaders
from src.ann_model import PlantDiseaseANN

def main():
    # 1. Load Data using your existing data_loader.py
    train_loader, val_loader, class_names = create_dataloaders()
    num_classes = len(class_names)
    
    # 2. Initialize the ANN
    # Calculate input size: Height * Width * Channels
    input_size = IMG_SIZE * IMG_SIZE * 3 
    model = PlantDiseaseANN(input_size, num_classes).to(DEVICE)
    
    print(f"\n🚀 Initializing ANN with {input_size} input features...")
    
    # 3. Define Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001) # Lower LR for large ANNs

    # 4. Training and Evaluation Loop
    for epoch in range(EPOCHS):
        # --- TRAINING ---
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()

        # --- EVALUATION ---
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {train_loss/len(train_loader):.4f} - Val Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()