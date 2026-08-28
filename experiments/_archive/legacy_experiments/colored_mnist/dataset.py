import torch
from torch.utils.data import Dataset
from torchvision.datasets import MNIST
import numpy as np

class ColoredMNIST(Dataset):
    def __init__(self, root: str, train: bool = True, download: bool = True, correlation: float = 0.9, seed: int = 42):
        """
        Colored MNIST dataset.
        
        Args:
            root (str): Root directory for MNIST data
            train (bool): If True, load training data, else test data
            download (bool): If True, download the dataset
            correlation (float): Probability that the color matches the label. 
                                 Set to 0.0 for test set to assign a random non-matching color.
            seed (int): Random seed for reproducibility of color assignments
        """
        super().__init__()
        self.mnist = MNIST(root=root, train=train, download=download)
        self.correlation = correlation
        
        # 10 distinct RGB colors normalized to [0, 1]
        self.colors = np.array([
            [1.0, 0.0, 0.0], # Red (0)
            [0.0, 1.0, 0.0], # Green (1)
            [0.0, 0.0, 1.0], # Blue (2)
            [1.0, 1.0, 0.0], # Yellow (3)
            [1.0, 0.0, 1.0], # Magenta (4)
            [0.0, 1.0, 1.0], # Cyan (5)
            [1.0, 0.5, 0.0], # Orange (6)
            [0.5, 0.0, 1.0], # Purple (7)
            [0.0, 1.0, 0.5], # Spring Green (8)
            [0.5, 0.5, 0.5]  # Gray (9)
        ], dtype=np.float32)
        
        # Pre-assign color indices to make the dataset fixed and reproducible
        rng = np.random.default_rng(seed)
        self.color_indices = []
        for i in range(len(self.mnist)):
            # Handle both tensor and integer targets
            target = self.mnist.targets[i]
            label = target.item() if hasattr(target, 'item') else int(target)
            
            if self.correlation == 0.0:
                # Test behavior: always non-matching color
                other_classes = [c for c in range(10) if c != label]
                color_idx = rng.choice(other_classes)
            else:
                if rng.random() < self.correlation:
                    color_idx = label
                else:
                    other_classes = [c for c in range(10) if c != label]
                    color_idx = rng.choice(other_classes)
            self.color_indices.append(color_idx)
            
    def __len__(self):
        return len(self.mnist)
        
    def __getitem__(self, idx):
        img, label = self.mnist[idx]
        img_arr = np.array(img, dtype=np.float32) / 255.0 # (28, 28)
        color_idx = self.color_indices[idx]
        color = self.colors[color_idx]
        
        # Apply color: shape (3, 28, 28)
        colored_img = np.zeros((3, 28, 28), dtype=np.float32)
        for c in range(3):
            colored_img[c, :, :] = img_arr * color[c]
            
        return torch.tensor(colored_img), label
