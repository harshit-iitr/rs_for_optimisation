import torch
import numpy as np
from torchvision.datasets import MNIST


class PermutedMNIST:
    """Permuted MNIST dataset generator for continual learning.

    Task 0 uses standard MNIST pixel ordering (or permuted if seed specifies).
    Tasks 1..n_tasks use fixed pseudo-random permutations of the 784 pixels.
    """

    def __init__(self, n_tasks=150, root="data", device="cpu", seed=1):
        self.n_tasks = n_tasks
        self.device = device
        self.seed = seed

        train_ds = MNIST(root=root, train=True, download=True)
        test_ds = MNIST(root=root, train=False, download=True)

        self.x_train = (train_ds.data.float() / 255.0).view(-1, 784).to(device)
        self.y_train = train_ds.targets.to(device)
        self.x_test = (test_ds.data.float() / 255.0).view(-1, 784).to(device)
        self.y_test = test_ds.targets.to(device)

        rng = np.random.default_rng(seed)
        self.permutations = [
            np.arange(784) if t == 0 else rng.permutation(784)
            for t in range(n_tasks)
        ]

    def get_task_data(self, task_idx):
        perm = self.permutations[task_idx]
        return (
            self.x_train[:, perm],
            self.y_train,
            self.x_test[:, perm],
            self.y_test,
        )
