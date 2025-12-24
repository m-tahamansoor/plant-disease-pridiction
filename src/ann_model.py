# src/ann_model.py
import torch.nn as nn

class PlantDiseaseANN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(PlantDiseaseANN, self).__init__()
        self.flatten = nn.Flatten()
        
        # Simple Multi-Layer Perceptron Architecture
        self.network = nn.Sequential(
            # Input layer: Flattened image (270,000) -> Hidden layer 1
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3), # Regularization to prevent overfitting
            
            # Hidden layer 2
            nn.Linear(512, 256),
            nn.ReLU(),
            
            # Output layer: Final prediction for 38 classes
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.network(x)