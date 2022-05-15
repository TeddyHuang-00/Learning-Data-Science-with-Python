import os

import torch
import torch.nn as nn


# LeNet-5 style model
class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Batch: N
        # CIFAR-10: 3 * 32 * 32
        self.conv1 = nn.Sequential(
            # N, 3, 32, 32 => N, 6, 32, 32
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True),
            # N, 6, 32, 32 => N, 6, 16, 16
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.conv2 = nn.Sequential(
            # N, 6, 16, 16 => N, 16, 12, 12
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            # N, 16, 12, 12 => N, 16, 6, 6
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.prediction = nn.Sequential(
            # N, 16, 6, 6 => N, 16 * 6 * 6
            nn.Flatten(),
            # N, 16 * 6 * 6 => N, 120
            nn.Linear(16 * 6 * 6, 120),
            nn.ReLU(inplace=True),
            # N, 120 => N, 84
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            # N, 84 => N, 10
            nn.Linear(84, num_classes),
        )

    def forward(self, X, *args):
        out = self.conv1(X)
        out = self.conv2(out)
        out = self.prediction(out)
        return out


if __name__ == "__main__":
    from time import time

    import pandas as pd
    from tqdm import tqdm

    from params import (
        ATTACK_ALPHA,
        ATTACK_EPS,
        ATTACK_ITER,
        BATCH_SIZE,
        LEARNING_RATE,
        NUM_EPOCHS,
    )
    from PGD import PGD
    from utils import DEVICE, Logger, get_train_test_dataloader

    # Hyperparameters
    num_epochs = NUM_EPOCHS
    learning_rate = LEARNING_RATE
    batch_size = BATCH_SIZE

    # Data
    train_loader, test_loader = get_train_test_dataloader(batch_size=batch_size)

    # Model
    model = CNN().to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    logger = Logger("./log/CNN.log")

    if not os.path.exists("./model/CNN.ckpt"):
        # Training
        logger(
            f"Epochs: {num_epochs}, Batch size: {batch_size}, Learning rate: {learning_rate}"
        )
        loss_history = []
        model.train()
        for epoch in range(num_epochs):
            start = time()
            for i, (images, labels) in enumerate(train_loader):
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                if (i + 1) % 100 == 0:
                    loss_history.append(loss.item())
                    logger(
                        f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}"
                    )
            now = time()
            logger(f"Epoch time elapsed: {now - start:.3f} sec")
        # Save the model checkpoint
        torch.save(model.state_dict(), "./model/CNN.ckpt")
        pd.DataFrame(loss_history).to_csv("./log/CNN_loss_history.csv")
    else:
        # Load the model checkpoint
        model.load_state_dict(torch.load("./model/CNN.ckpt", map_location=DEVICE))

    # Testing
    total = 0
    correct = 0
    correct_adv = 0
    success = 0
    attacker = PGD(eps=ATTACK_EPS, alpha=ATTACK_ALPHA, num_iter=ATTACK_ITER)
    model.eval()

    for images, labels in tqdm(test_loader):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        total += labels.size(0)
        # Model evaluation
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        # Attack
        images_adv = attacker.attack(model, images, labels)
        outputs = model(images_adv)
        _, predicted_adv = torch.max(outputs.data, 1)
        correct_adv += (predicted_adv == labels).sum().item()
        success -= (
            torch.where(predicted == labels, predicted_adv == labels, False)
            .sum()
            .item()
        )
    success += correct
    logger(f"Tested on {total} samples.")
    logger(
        f"Attack amplitude: {ATTACK_EPS:.3f}, step size: {ATTACK_ALPHA:.3f}, iterations: {ATTACK_ITER:d}"
    )
    logger(f"Original accuracy : {100 * correct / total:.2f}%")
    logger(f"Adversarial accuracy : {100 * correct_adv / total:.2f}%")
    logger(f"Adversarial success rate : {100 * success / correct:.2f}%")
