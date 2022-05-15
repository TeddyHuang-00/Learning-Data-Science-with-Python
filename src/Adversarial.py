import os

import torch
import torch.nn as nn


class AdvTrain(nn.Module):
    def __init__(self, model, attacker) -> None:
        super().__init__()
        self.model = model
        self.attacker = attacker

    def forward(self, X, y=None):
        # Only modify the input during training steps
        if self.training:
            # Need to get the correct labels for loss calculation
            assert y is not None
            # Get attack images
            X_adv = self.attacker.attack(self.model, X, y)
            # Feed to model
            return self.model(X_adv)
        else:
            return self.model(X)


if __name__ == "__main__":
    from time import time

    import pandas as pd
    from tqdm import tqdm

    from CNN import CNN
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
    attacker = PGD(eps=ATTACK_EPS, alpha=ATTACK_ALPHA, num_iter=ATTACK_ITER)
    trainer = AdvTrain(model, attacker)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    logger = Logger("./log/Adv.log")

    if not os.path.exists("./model/Adv.ckpt"):
        # Train
        logger(
            f"Epochs: {num_epochs}, Batch size: {batch_size}, Learning rate: {learning_rate}"
        )
        loss_history = []
        trainer.train()
        for epoch in range(num_epochs):
            start = time()
            for i, (X, y) in enumerate(train_loader):
                X = X.to(DEVICE)
                y = y.to(DEVICE)
                optimizer.zero_grad()
                y_pred = trainer(X, y)
                loss = criterion(y_pred, y)
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
        torch.save(trainer.state_dict(), "./model/Adv.ckpt")
        pd.DataFrame(loss_history).to_csv("./log/Adv_loss_history.csv")
    else:
        # Load the model checkpoint
        model.load_state_dict(torch.load("./model/Adv.ckpt", map_location=DEVICE))

    # Testing
    total = 0
    correct = 0
    correct_adv = 0
    success = 0
    trainer.eval()

    for images, labels in tqdm(test_loader):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        total += labels.size(0)
        outputs = trainer(images)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        images_adv = attacker.attack(model, images, labels)
        outputs = trainer(images_adv)
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
