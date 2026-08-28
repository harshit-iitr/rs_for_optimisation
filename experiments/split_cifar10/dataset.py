import torch
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import TensorDataset

class SplitCIFAR10Loader:
    def __init__(self, root: str, train: bool = True, download: bool = True):
        """
        Helper class to load CIFAR-10, split into tasks, and hold in memory.
        """
        # CIFAR-10 transform: convert to tensor and normalize
        # We don't use data augmentation to isolate plasticity dynamics clearly
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        
        # Load standard dataset
        self.cifar = CIFAR10(root=root, train=train, download=download, transform=transform)
        
        # Load all images and labels into RAM
        # Using DataLoader is slow, so we can convert the entire dataset to tensors
        images_list = []
        labels_list = []
        for img, label in self.cifar:
            images_list.append(img.unsqueeze(0))
            labels_list.append(label)
            
        self.images = torch.cat(images_list, dim=0) # (N, 3, 32, 32)
        self.labels = torch.tensor(labels_list, dtype=torch.long) # (N,)

    def get_task_dataset(self, task_idx: int) -> TensorDataset:
        """
        Returns a TensorDataset for the specified task index (0 to 4).
        Each task contains 2 classes: class 2*task_idx and 2*task_idx + 1.
        Labels are mapped to 0 and 1.
        """
        class_a = 2 * task_idx
        class_b = 2 * task_idx + 1
        
        mask = (self.labels == class_a) | (self.labels == class_b)
        task_images = self.images[mask]
        task_labels = self.labels[mask]
        
        # Map labels to 0 and 1
        mapped_labels = torch.where(task_labels == class_a, 0, 1)
        
        return TensorDataset(task_images, mapped_labels)
