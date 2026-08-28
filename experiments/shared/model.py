import torch
import torch.nn as nn
from typing import Tuple, List

class SharedMLP(nn.Module):
    def __init__(self, d_in: int, d_out: int, d_hidden: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_hidden)
        self.fc3 = nn.Linear(d_hidden, d_out)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, d_in)
            
        Returns:
            logits (torch.Tensor): Output logits of shape (batch_size, d_out)
            pre_acts (List[torch.Tensor]): List of pre-activation outputs of hidden layers, [h1, h2]
        """
        h1 = self.fc1(x)
        a1 = self.relu(h1)
        
        h2 = self.fc2(a1)
        a2 = self.relu(h2)
        
        logits = self.fc3(a2)
        return logits, [h1, h2]


class DeepMLP(nn.Module):
    def __init__(self, d_in: int, d_out: int, d_hidden: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_hidden)
        self.fc3 = nn.Linear(d_hidden, d_hidden)
        self.fc4 = nn.Linear(d_hidden, d_out)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass for a 3-hidden-layer MLP.
        """
        h1 = self.fc1(x)
        a1 = self.relu(h1)
        
        h2 = self.fc2(a1)
        a2 = self.relu(h2)
        
        h3 = self.fc3(a2)
        a3 = self.relu(h3)
        
        logits = self.fc4(a3)
        return logits, [h1, h2, h3]


class SimpleCNN(nn.Module):
    def __init__(self, num_classes_per_head: int = 2, num_tasks: int = 5, d_hidden: int = 256):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        
        # Flatten size for 32x32 image with 3 pools is 128 * 4 * 4 = 2048
        self.fc_hidden = nn.Linear(2048, d_hidden)
        
        # Multi-head classification layers
        self.heads = nn.ModuleList([
            nn.Linear(d_hidden, num_classes_per_head) for _ in range(num_tasks)
        ])

    def forward(self, x: torch.Tensor, task_idx: int = 0) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass for CNN.
        
        Args:
            x (torch.Tensor): Batch of images of shape (B, 3, 32, 32)
            task_idx (int): The task ID to route the final classification head
            
        Returns:
            logits (torch.Tensor): Classification logits (B, num_classes_per_head)
            pre_acts (List[torch.Tensor]): [h_conv1, h_conv2, h_conv3, h_fc]
        """
        h_conv1 = self.conv1(x)
        x = self.pool(self.relu(h_conv1))
        
        h_conv2 = self.conv2(x)
        x = self.pool(self.relu(h_conv2))
        
        h_conv3 = self.conv3(x)
        x = self.pool(self.relu(h_conv3))
        
        x = x.view(x.shape[0], -1) # Flatten
        
        h_fc = self.fc_hidden(x)
        a_fc = self.relu(h_fc)
        
        logits = self.heads[task_idx](a_fc)
        return logits, [h_conv1, h_conv2, h_conv3, h_fc]

