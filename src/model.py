# src/model.py

import torch.nn as nn
from torchvision import models

class PlantDiseaseClassifier(nn.Module):
    def __init__(self, num_classes):
        super(PlantDiseaseClassifier, self).__init__()
        
        # 1. Load pre-trained MobileNetV2
        self.base_model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        
        # 2. Freeze the base model layers
        for param in self.base_model.parameters():
            param.requires_grad = False

        # 3. Define the custom classification head
        # MobileNetV2's final classification layer is stored in base_model.classifier[1]
        in_features = self.base_model.classifier[1].in_features
        
        self.classifier_head = nn.Sequential(
            nn.Dropout(0.2), # Standard dropout before the classification layer
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
        # Replace the original classifier with the custom head
        self.base_model.classifier = self.classifier_head
        
    def forward(self, x):
        return self.base_model(x)

def build_model(num_classes, device):
    """
    Initializes and returns the PyTorch model, moved to the specified device.
    """
    model = PlantDiseaseClassifier(num_classes=num_classes)
    model.to(device)
    print("✅ PyTorch MobileNetV2 model initialized.")
    return model