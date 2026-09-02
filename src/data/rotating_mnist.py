import torch
import torchvision.transforms.functional as TF
from torchvision.datasets import MNIST


class RotatingMNIST:
    """Rotating MNIST dataset generator for continual learning.

    Rotates images continuously from 0 to 180 degrees across tasks.
    """

    def __init__(self, n_tasks=100, root="data", device="cpu", seed=1):
        self.n_tasks = n_tasks
        self.device = device
        self.seed = seed

        train_ds = MNIST(root=root, train=True, download=True)
        test_ds = MNIST(root=root, train=False, download=True)

        self.x_train_imgs = (train_ds.data.float() / 255.0).unsqueeze(1)
        self.y_train = train_ds.targets.to(device)
        self.x_test_imgs = (test_ds.data.float() / 255.0).unsqueeze(1)
        self.y_test = test_ds.targets.to(device)

        self.angles = [t * (180.0 / max(1, n_tasks - 1)) for t in range(n_tasks)]

    def get_task_data(self, task_idx):
        angle = self.angles[task_idx]
        if angle == 0:
            tr = self.x_train_imgs.view(-1, 784).to(self.device)
            te = self.x_test_imgs.view(-1, 784).to(self.device)
        else:
            tr = TF.rotate(self.x_train_imgs, angle).view(-1, 784).to(self.device)
            te = TF.rotate(self.x_test_imgs, angle).view(-1, 784).to(self.device)
        return tr, self.y_train, te, self.y_test
