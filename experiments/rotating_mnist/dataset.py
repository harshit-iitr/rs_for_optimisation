import torch
from torchvision.datasets import MNIST
import numpy as np

class RotatingMNISTLoader:
    def __init__(self, root: str, train: bool = True, download: bool = True):
        """
        Helper class to load and hold MNIST in memory for fast on-the-fly rotation.
        """
        self.mnist = MNIST(root=root, train=train, download=download)
        # Convert to float and normalize to [0, 1]
        self.images = self.mnist.data.float() / 255.0 # (N, 28, 28)
        self.images = self.images.unsqueeze(1) # (N, 1, 28, 28) for rotation functions
        
        targets = self.mnist.targets
        if not isinstance(targets, torch.Tensor):
            targets = torch.tensor(targets)
        self.labels = targets.long()
