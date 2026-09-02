import torch
import torchvision
from torchvision import transforms
import numpy as np

class PermutedMNIST:
    def __init__(self, root='./data', n_tasks=150, device='cuda', seed=42):
        self.n_tasks = n_tasks
        self.device = device
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        train_dataset = torchvision.datasets.MNIST(root=root, train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.MNIST(root=root, train=False, download=True, transform=transform)
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
        
        for x, y in train_loader:
            self.x_train = x.view(len(x), -1).to(device)
            self.y_train = y.to(device)
            break
            
        for x, y in test_loader:
            self.x_test = x.view(len(x), -1).to(device)
            self.y_test = y.to(device)
            break
            
        rng = np.random.default_rng(seed)
        self.permutations = []
        for _ in range(n_tasks):
            perm = torch.from_numpy(rng.permutation(784)).to(device)
            self.permutations.append(perm)
            
    def get_task_data(self, task_idx):
        """Returns pre-permuted full train and test tensors."""
        perm = self.permutations[task_idx]
        return (
            self.x_train[:, perm], self.y_train,
            self.x_test[:, perm], self.y_test
        )
