import torch
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF

class RotatingMNIST:
    def __init__(self, n_tasks=100, device='cpu', seed=42):
        self.n_tasks = n_tasks
        self.device = device
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)
        
        self.x_train_base = train_dataset.data.float().unsqueeze(1) / 255.0
        self.x_train_base = (self.x_train_base - 0.1307) / 0.3081
        self.y_train = train_dataset.targets.to(device)
        
        self.x_test_base = test_dataset.data.float().unsqueeze(1) / 255.0
        self.x_test_base = (self.x_test_base - 0.1307) / 0.3081
        self.y_test = test_dataset.targets.to(device)
        
    def get_task_data(self, task_idx):
        angle = task_idx * (180.0 / max(1, self.n_tasks - 1))
        
        x_train_rot = TF.rotate(self.x_train_base, angle)
        x_test_rot = TF.rotate(self.x_test_base, angle)
        
        return x_train_rot.to(self.device), self.y_train, x_test_rot.to(self.device), self.y_test
