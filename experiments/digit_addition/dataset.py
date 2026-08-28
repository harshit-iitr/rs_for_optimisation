import torch
from torchvision.datasets import MNIST
import numpy as np
from torch.utils.data import Dataset

class MNISTAdditionDataset(Dataset):
    def __init__(self, root: str, train: bool = True, download: bool = True, f_train: float = 0.3, num_pairs: int = 10000, seed: int = 42):
        """
        Dataset that loads MNIST and creates digit addition pairs.
        
        Args:
            root (str): Root directory for MNIST
            train (bool): If True, use training images and training equations.
            download (bool): If True, download dataset.
            f_train (float): Fraction of the 100 possible modular addition equations to use for training.
            num_pairs (int): Total number of pair samples to generate.
            seed (int): Random seed for reproducibility.
        """
        super().__init__()
        # Ensure we load correct MNIST subset (train or test)
        self.mnist = MNIST(root=root, train=train, download=download)
        
        # Group MNIST image indices by digit
        self.digit_to_indices = {i: [] for i in range(10)}
        for idx in range(len(self.mnist)):
            target = self.mnist.targets[idx]
            label = target.item() if hasattr(target, 'item') else int(target)
            self.digit_to_indices[label].append(idx)
            
        # Convert to numpy arrays for extremely fast rng.choice
        for i in range(10):
            self.digit_to_indices[i] = np.array(self.digit_to_indices[i], dtype=np.int64)
            
        # Select train equations (e.g. 30% of the 100 possible a+b pairs)
        rng = np.random.default_rng(seed)
        all_pairs = [(a, b) for a in range(10) for b in range(10)]
        rng.shuffle(all_pairs)
        
        split_idx = int(len(all_pairs) * f_train)
        train_pairs = set(all_pairs[:split_idx])
        test_pairs = set(all_pairs[split_idx:])
        
        target_pairs = train_pairs if train else test_pairs
        self.target_pairs_list = list(target_pairs)
        
        # Build samples deterministically
        self.samples = []
        for _ in range(num_pairs):
            # Choose an equation from the set
            eq_idx = rng.choice(len(self.target_pairs_list))
            a, b = self.target_pairs_list[eq_idx]
            
            # Choose random images of these digits
            idx_a = rng.choice(self.digit_to_indices[a])
            idx_b = rng.choice(self.digit_to_indices[b])
            
            self.samples.append((idx_a, idx_b, (a + b) % 10))
            
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        idx_a, idx_b, target_sum = self.samples[idx]
        
        img_a, _ = self.mnist[idx_a]
        img_b, _ = self.mnist[idx_b]
        
        # Convert images to numpy arrays and normalize
        arr_a = np.array(img_a, dtype=np.float32) / 255.0
        arr_b = np.array(img_b, dtype=np.float32) / 255.0
        
        # Concatenate images side by side: (28, 56)
        concat_img = np.hstack((arr_a, arr_b))
        
        # Flatten to 1568-dim vector
        flat_img = concat_img.flatten()
        
        return torch.tensor(flat_img, dtype=torch.float32), target_sum
