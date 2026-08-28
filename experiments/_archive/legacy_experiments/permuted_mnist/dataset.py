import torch
from torchvision.datasets import MNIST
import numpy as np
from torch.utils.data import TensorDataset

class PermutedMNISTLoader:
    def __init__(self, root: str, train: bool = True, download: bool = True):
        """
        Helper class to load and hold MNIST in memory, facilitating fast permutation.
        """
        self.mnist = MNIST(root=root, train=train, download=download)
        # Convert to float and normalize to [0, 1]
        self.images = self.mnist.data.float() / 255.0 # (N, 28, 28)
        self.images = self.images.view(-1, 784) # (N, 784)
        
        # Handle both tensor and list targets
        targets = self.mnist.targets
        if not isinstance(targets, torch.Tensor):
            targets = torch.tensor(targets)
        self.labels = targets.long()

    def get_task_dataset(self, permutation: np.ndarray) -> TensorDataset:
        """
        Returns a TensorDataset where the pixels are permuted according to the given permutation array.
        """
        # Permute columns of the image matrix
        permuted_images = self.images[:, permutation]
        return TensorDataset(permuted_images, self.labels)

def get_permutations(seed: int = 1234, num_tasks: int = 10) -> list:
    """
    Generates a list of fixed random permutations of length 784.
    """
    rng = np.random.default_rng(seed)
    permutations = []
    for _ in range(num_tasks):
        perm = rng.permutation(784)
        permutations.append(perm)
    return permutations
