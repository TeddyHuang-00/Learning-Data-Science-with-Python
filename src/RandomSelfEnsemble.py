import os

import torch
import torch.nn as nn


class RSE(nn.Module):
    def __init__(self, model, init_level=0.1, inner_level=0.05, ensemble_num=5) -> None:
        super().__init__()
        self.model = model
        self.init_level = init_level
        self.inner_level = inner_level
        self.ensemble_num = ensemble_num

    def forward(self, X):
        # We only go through forward pass once when training
        if self.training:
            first_flag = True
            out = X
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Sequential):
                    if "conv" in name:
                        # Add different levels of noise to the layer inputs
                        if first_flag:
                            out = module(
                                out
                                + torch.normal(
                                    mean=0, std=self.init_level, size=out.shape
                                ).to(out.device)
                            )
                            first_flag = False
                        else:
                            out = module(
                                out
                                + torch.normal(
                                    mean=0, std=self.inner_level, size=out.shape
                                ).to(out.device)
                            )
                    else:
                        out = module(out)
            return out
        # We would need to go through standard forward pass multiple times
        # for the ensemble part
        else:
            total = 0
            for _ in range(self.ensemble_num):
                first_flag = True
                out = X
                for name, module in self.model.named_modules():
                    if isinstance(module, nn.Sequential):
                        if "conv" in name:
                            if first_flag:
                                out = module(
                                    out
                                    + torch.normal(
                                        mean=0, std=self.init_level, size=out.shape
                                    ).to(out.device)
                                )
                                first_flag = False
                            else:
                                out = module(
                                    out
                                    + torch.normal(
                                        mean=0, std=self.inner_level, size=out.shape
                                    ).to(out.device)
                                )
                        else:
                            out = module(out)
                total = total + out
            return total / self.ensemble_num


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
        RSE_INIT,
        RSE_INNER,
        RSE_NUM,
    )
    from PGD import PGD
    from utils import DEVICE, Logger, get_train_test_dataloader

    # Hyperparameters
    num_epochs = NUM_EPOCHS
    learning_rate = LEARNING_RATE
    batch_size = BATCH_SIZE
    init_level = RSE_INIT
    inner_level = RSE_INNER
    ensemble_num = RSE_NUM

    # Data
    train_loader, test_loader = get_train_test_dataloader(batch_size=batch_size)

    # Model
    model = CNN().to(DEVICE)
    trainer = RSE(
        model, init_level=init_level, inner_level=inner_level, ensemble_num=ensemble_num
    ).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(trainer.parameters(), lr=learning_rate, momentum=0.9)
    logger = Logger("./log/RSE.log")

    if not os.path.exists("./model/RSE.ckpt"):
        # Train
        logger(
            f"Epochs: {num_epochs}, Batch size: {batch_size}, Learning rate: {learning_rate}, "
            f"Init noise level: {init_level}, Inner noise level: {inner_level}, Ensemble number: {ensemble_num}"
        )
        loss_history = []
        trainer.train()
        for epoch in range(num_epochs):
            start = time()
            for i, (X, y) in enumerate(train_loader):
                X = X.to(DEVICE)
                y = y.to(DEVICE)
                optimizer.zero_grad()
                y_pred = trainer(X)
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
        torch.save(trainer.state_dict(), "./model/RSE.ckpt")
        pd.DataFrame(loss_history).to_csv("./log/RSE_loss_history.csv")
    else:
        # Load the model checkpoint
        trainer.load_state_dict(torch.load("./model/RSE.ckpt", map_location=DEVICE))

    # Testing
    total = 0
    correct = 0
    correct_adv = 0
    success = 0
    trainer.eval()
    attacker = PGD(eps=ATTACK_EPS, alpha=ATTACK_ALPHA, num_iter=ATTACK_ITER)

    for images, labels in tqdm(test_loader):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        total += labels.size(0)
        outputs = trainer(images)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        images_adv = attacker.attack(trainer, images, labels)
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
