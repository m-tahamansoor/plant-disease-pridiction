# src/config.py

import os
import torch
from pathlib import Path

# --- GLOBAL CONFIGURATION ---
BATCH_SIZE = 32
IMG_SIZE = 300
EPOCHS = 20

# --- DEVICE CONFIGURATION ---
# Check for CUDA availability and set the device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- PATHS (Updated for your directory structure) ---
BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_FOLDER_NAME = 'New Plant Diseases Dataset(Augmented)' 
DATA_PATH = BASE_DIR / 'data' / DATASET_FOLDER_NAME

TRAIN_PATH = DATA_PATH / 'train'
VALID_PATH = DATA_PATH / 'valid'

# --- OUTPUT PATHS ---
MODELS_DIR = BASE_DIR / 'models'
MODELS_DIR.mkdir(exist_ok=True) 

CHECKPOINT_PATH = MODELS_DIR / 'best_model.pth' 
LABELS_PATH = MODELS_DIR / 'labels.txt'

# Assert paths exist for safety
if not TRAIN_PATH.is_dir() or not VALID_PATH.is_dir():
    raise FileNotFoundError(f"Data directories not found. Please ensure the dataset is at: {DATA_PATH}")

# --- FIX APPLIED HERE: Only print once in the main process ---
# This check prevents worker processes from re-running the print statements.
if __name__ == '__main__':
    print(f"Using device: {DEVICE}")
    print(f"Base Directory: {BASE_DIR}")
    print(f"Data Path: {DATA_PATH}")

# --- VISUALIZATION CONFIG ---
# For EfficientNetB3, the last convolutional layer is in the 'features' block at index 8
LAST_CONV_LAYER_NAME = 'features.8'