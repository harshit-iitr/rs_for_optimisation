import torch
from torchvision import datasets, transforms

class SplitCIFAR100:
    def __init__(self, n_tasks=10, device='cpu', seed=42):
        self.n_tasks = n_tasks
        self.device = device
        self.classes_per_task = 100 // n_tasks
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        
        train_dataset = datasets.CIFAR100('data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR100('data', train=False, download=True, transform=transform)
        
        # We need to load all data to GPU if possible, but CIFAR100 might be a bit large.
        # It's 50000x3x32x32 floats = ~600MB, easily fits in GPU memory.
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=50000, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10000, shuffle=False)
        
        self.x_train_all, self.y_train_all = next(iter(train_loader))
        self.x_test_all, self.y_test_all = next(iter(test_loader))
        
        self.x_train_all = self.x_train_all.to(device)
        self.y_train_all = self.y_train_all.to(device)
        self.x_test_all = self.x_test_all.to(device)
        self.y_test_all = self.y_test_all.to(device)
        
    def get_task_data(self, task_idx):
        start_class = task_idx * self.classes_per_task
        end_class = start_class + self.classes_per_task
        
        train_mask = (self.y_train_all >= start_class) & (self.y_train_all < end_class)
        test_mask = (self.y_test_all >= start_class) & (self.y_test_all < end_class)
        
        return self.x_train_all[train_mask], self.y_train_all[train_mask], self.x_test_all[test_mask], self.y_test_all[test_mask]
