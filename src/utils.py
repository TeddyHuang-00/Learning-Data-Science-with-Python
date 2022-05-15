import os

import numpy as np
import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms

from params import SEED

# Set seed
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

# Check folders
for folder in ["data", "model", "log", "fig"]:
    if not os.path.exists(f"./{folder}"):
        os.mkdir(f"./{folder}")

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data
MEAN = (0.4942, 0.4851, 0.4504)
STD = (0.2467, 0.2429, 0.2616)
tf = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ]
)
# the following transforms are used for data augmentation
# but here to keep it simple, we don't actually use them
# it is reasonable to swap the default transforms with corresponding DA ones.
train_tf = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        tf,
    ]
)
test_tf = transforms.Compose([tf])

class_labels = [
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

if not os.path.exists("./data/cifar-10-python.tar.gz"):
    # Just download the data if not already exists and omit the notification
    _ = torchvision.datasets.CIFAR10(root="./data", download=True, transform=tf)
train_set = torchvision.datasets.CIFAR10(root="./data", train=True, transform=tf)
test_set = torchvision.datasets.CIFAR10(root="./data", train=False, transform=tf)


def get_train_dataloader(batch_size, shuffle=True):
    global train_set
    return data.DataLoader(
        train_set, batch_size=batch_size, shuffle=shuffle, num_workers=0
    )


def get_test_dataloader(batch_size, shuffle=False):
    global test_set
    return data.DataLoader(
        test_set, batch_size=batch_size, shuffle=shuffle, num_workers=0
    )


def get_train_test_dataloader(batch_size):
    global train_set, test_set
    return get_train_dataloader(batch_size), get_test_dataloader(batch_size)


# Logger
class Logger:
    """
    Takes a file path when initialized

    print and write to file for anything passed to it
    """

    def __init__(self, file_path) -> None:
        self.file_path = file_path
        print(f"{DEVICE=}, Pytorch: {torch.__version__}, Seed: {SEED}")
        with open(file_path, "w") as f:
            f.write(f"{DEVICE=}, Pytorch: {torch.__version__}, Seed: {SEED}\n")

    def __call__(self, message) -> None:
        print(message)
        with open(self.file_path, "a") as f:
            f.write(message + "\n")


if __name__ == "__main__":
    print(f"{DEVICE=}, Pytorch: {torch.__version__}, Seed: {SEED}")
    print(f"{len(train_set)} train samples, {len(test_set)} test samples")
    # Only used to check if data transformation is correct
    print(
        f"Test mean: {next(iter(get_test_dataloader(len(test_set))))[0].mean(dim=(0,2,3))}"
    )
    print(
        f"Test std: {next(iter(get_test_dataloader(len(test_set))))[0].std(dim=(0,2,3))}"
    )
    print(
        f"Train mean: {next(iter(get_test_dataloader(len(train_set))))[0].mean(dim=(0,2,3))}"
    )
    print(
        f"Train std: {next(iter(get_test_dataloader(len(train_set))))[0].std(dim=(0,2,3))}"
    )
