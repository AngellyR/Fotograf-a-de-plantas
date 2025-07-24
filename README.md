# 🌿 Plant Identifier - Plant Image Classification with Deep Learning

This project trains a convolutional neural network to classify plant species using online images sourced from a CSV file. It follows a clean modular structure for data preparation, training, and model evaluation.

## Setup Instructions

### 1. Environment

```bash
# Create a new conda environment named 'plants'
conda create -n plants python=3.13

# Activate the environment
conda activate plants
```

### 2. Install Dependencies

```bash
# Path of the folder to run
cd "C:\Users\angyp\Documents\anaconda_projects\Fotografia de Plantas\Mod" 

# Install required packages
pip install -r requirements.txt
```

### 3. Launch Jupyter Notebook

```bash
# Start Jupyter Notebook
jupyter notebook
```

Then open `replic.ipynb` in your browser.

## 📁 Projects Structure

```bash

MOD/
├── data/
│   ├── raw/
│   │   ├── Dataset_plants/        # Original GBIF TXT/XML dataset
│   │   │   ├── dataset/
│   │   │   │    └── ...           #  XML dataset
│   │   │   ├── `citation.txt`
│   │   │   ├── `meta.xml`
│   │   │   ├── `metadat.xml`
│   │   │   ├── `multimedia.txt`
│   │   │   ├── `occurrence.txt`
│   │   │   ├── `rights.txt`
│   │   │   └── `verbatim.txt`
│   │   └── `merged.csv`           # GBIF merged dataset with columns specifics
│   └── `url_labels.csv`           # Cleaned dataset with URLs and label IDs
│
├── helpers/
│   ├── _pycache_/
│   ├── `dataset_local.py`         # Class LocalPlantDataset for local images
│   └── `train_utils.py`           # Training function with metrics/logging
│
├── models/                        # Trained PyTorch models are saved here
│   ├── `alexnet.pth`
│   └── `efficientnet_b3.pth`
│
├── notebooks/
│   ├── `download_dataset.ipynb`   # Download the dataset of ``url_labels.csv``
│   ├── `preprocess_data.ipynb`    # Cleans `merged.csv` into `url_labels.csv`
│   ├── `testing.ipynb`            # Models with random images are tested
│   ├── `train_alexnet.ipynb`      # Loads data, trains AlexNet and plots metrics
│   └── `train_efficientnet.ipynb` # Loads data, trains EfficientNet and plots metrics
│
├── `README.md`                    # Project documentation
└── `requirements.txt`             # Python dependencies
```

## How to train

### 1. Preprocess the dataset and download

```bash
# Run
notebooks/preprocess_data.ipynb
```

It will create `url_labels.csv` with:

- url: valid image URL (from GBIF identifier or references)
- scientificName
- label_id: integer-encoded label

```bash
# Run
notebooks/download_dataset
```

Then it will download the dataset. Once the dataset is downloaded, folders containing fewer than 100 images will be removed, and only those with at least 100 images will be kept.

### 2. Train Models

#### 2.1. AlexNet

```bash
# Run
notebooks/train_alexnet.ipynb
```

- Loads local images from data/images/
- Trains AlexNet with accuracy and loss logs per epoch
- Saves model to `models/alexnet.pth`

#### 2.2. EfficientNetV2-S

```bash
# Run
notebooks/train_efficientnet.ipynb
```

- Uses the same dataset structure
- Trains EfficientNetV2-S using torchvision.models.efficientnet_v2_s
- Saves model to `models/efficientnet_b3.pth`

### 3. Evaluate

Accuracy and loss graphs are shown automatically after each training.

## Requirements

```bash
# Install dependencies:
pip install -r requirements.txt
```

## Note

- Models can be extended easily with ResNet, VGG, etc.
- EfficientNet tends to perform better on small datasets with fewer parameters.
- Recommended: Use a GPU to accelerate training.
- Sources who maybe would be useful [Deep-Plant GitHub Repository](https://github.com/cs-chan/Deep-Plant)
