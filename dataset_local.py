# Import required libraries
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from collections import defaultdict

class LocalPlantDataset(Dataset):
    def __init__(self, root_dir, transform=None, min_samples=0):
        """
        Args:
            root_dir (string): Base directory for images
            transform (callable, optional): Transformations to apply
            min_samples (int): Minimum number of images per class (0 = no filter)
        """
        self.root_dir = root_dir
        self.transform = transform
        self.min_samples = min_samples
        
        # Load initial classes
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # Build dataset
        self.samples = self._make_dataset()
        self.targets = [label for _, label in self.samples]
        
        # Apply filter if necessary
        if min_samples > 0:
            self._filter_classes(min_samples)
        
    def _make_dataset(self):
        samples = []
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            if os.path.isdir(class_dir):
                class_idx = self.class_to_idx[class_name]
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(class_dir, img_name)
                        samples.append((img_path, class_idx))
        return samples
        
    def _filter_classes(self, min_samples):
        # Count samples per class
        class_counts = defaultdict(int)
        for _, label in self.samples:
            class_counts[label] += 1
        
        # Identify valid classes
        valid_classes = [cls for cls, count in class_counts.items() if count >= min_samples]
        
        # Create new class mapping
        new_classes = []
        new_class_to_idx = {}
        new_samples = []
        
        # Reassign new consecutive indexes
        for new_idx, old_idx in enumerate(valid_classes):
            class_name = self.classes[old_idx]
            new_classes.append(class_name)
            new_class_to_idx[class_name] = new_idx
        
        # Update samples with new indexes
        for path, old_label in self.samples:
            if old_label in valid_classes:
                class_name = self.classes[old_label]
                new_label = new_class_to_idx[class_name]
                new_samples.append((path, new_label))
        
        # Update class variables
        self.samples = new_samples
        self.classes = new_classes
        self.class_to_idx = new_class_to_idx
        self.targets = [label for _, label in self.samples]
        
        print(f"Classes after filtering: {len(self.classes)}")
        print(f"Samples after filtering: {len(self.samples)}")
            
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            # Fallback: blank image
            blank_image = Image.new('RGB', (224, 224), (255, 255, 255))
            if self.transform:
                blank_image = self.transform(blank_image)
            return blank_image, label