import torch
import torchvision
from torchvision import transforms

class SplitMNIST:
    """Split MNIST benchmark for Continual Learning (CIL, TIL, and DIL).
    
    Divides MNIST into 5 tasks of 2 digits each:
    Task 0: digits 0, 1
    Task 1: digits 2, 3
    Task 2: digits 4, 5
    Task 3: digits 6, 7
    Task 4: digits 8, 9
    
    Supports:
    - Class-Incremental (CIL): Default setting. Output dimension is 10.
    - Task-Incremental (TIL): Use CIL setting, but pass `--task_incremental` to training script.
    - Domain-Incremental (DIL): Set `domain_incremental=True`. Targets are remapped to 0 and 1.
    """
    def __init__(self, root='./data', n_tasks=5, device='cuda', seed=42, domain_incremental=False):
        # We enforce exactly 5 tasks for Split MNIST
        if n_tasks != 5:
            print(f"Warning: SplitMNIST naturally has 5 tasks. Overriding n_tasks={n_tasks} to 5.")
        self.n_tasks = 5
        self.device = device
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        train_dataset = torchvision.datasets.MNIST(root=root, train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.MNIST(root=root, train=False, download=True, transform=transform)
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
        
        # Load all data into memory
        for x, y in train_loader:
            self.x_train = x.view(len(x), -1).to(device)
            self.y_train = y.to(device)
            break
            
        for x, y in test_loader:
            self.x_test = x.view(len(x), -1).to(device)
            self.y_test = y.to(device)
            break
            
        # Pre-filter indices for each task
        self.task_data = []
        for t in range(5):
            c1, c2 = 2 * t, 2 * t + 1
            
            # Find indices where targets are c1 or c2
            train_mask = (self.y_train == c1) | (self.y_train == c2)
            test_mask = (self.y_test == c1) | (self.y_test == c2)
            
            ytrain = self.y_train[train_mask].clone()
            ytest = self.y_test[test_mask].clone()
            
            if domain_incremental:
                # Remap labels to 0 and 1
                ytrain[ytrain == c1] = 0
                ytrain[ytrain == c2] = 1
                ytest[ytest == c1] = 0
                ytest[ytest == c2] = 1
            
            self.task_data.append((
                self.x_train[train_mask], ytrain,
                self.x_test[test_mask], ytest
            ))
            
    def get_task_data(self, task_idx):
        """Returns the pre-filtered train and test tensors for the task."""
        if task_idx >= 5:
            raise ValueError(f"SplitMNIST only supports 5 tasks (0-4). Received task_idx={task_idx}")
        return self.task_data[task_idx]
