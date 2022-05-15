import os
from abc import abstractmethod

import torch
import torch.nn as nn
from torch.nn import functional as F


class Regularization(nn.Module):
    def __init__(self, lambda_):
        super().__init__()
        assert lambda_ > 0
        self.lambda_ = lambda_

    @staticmethod
    def _get_weight_list(model):
        # Actually returns a generator
        return (
            (name, param)
            for name, param in model.named_parameters()
            if "weight" in name and param.requires_grad and param.dim() in (2, 4)
        )

    def print_weight_list(self, model):
        weight_list = self._get_weight_list(model)
        for name, param in weight_list:
            print(name, param.shape)

    @abstractmethod
    def forward(self, model):
        ...


class SpectralNorm(Regularization):
    def forward(self, model):
        loss = 0
        for name, param in self._get_weight_list(model):
            mat_W = param
            if mat_W.dim() == 4:
                # For convolutional layers
                # I know this is shit code, you should never do this for compatibility reasons
                h_in, w_in, kernel_size, pad, in_channels, out_channels = (
                    (32, 32, 5, 2, 3, 6) if "conv1" in name else (16, 16, 5, 0, 6, 16)
                )
                h_out = h_in - kernel_size + 1 + 2 * pad
                w_out = w_in - kernel_size + 1 + 2 * pad
                u, v = (
                    torch.ones(out_channels, h_out, w_out, device=mat_W.device),
                    torch.ones(in_channels, h_in, w_in, device=mat_W.device),
                )
                for _ in range(2):
                    u, v = (
                        F.conv2d(v, mat_W, bias=None, stride=1, padding=pad),
                        F.conv2d(
                            u,
                            mat_W.transpose(0, 1).flip(2, 3),
                            bias=None,
                            stride=1,
                            padding=kernel_size - 1 - pad,
                        ),
                    )
                    u, v = (u / torch.norm(u), v / torch.norm(v))
                s = (F.conv2d(v, mat_W, bias=None, stride=1, padding=pad) * u).sum()
            else:
                # For linear layers
                h, w = mat_W.shape
                u, v = (
                    torch.ones(h, requires_grad=False).to(mat_W.device),
                    torch.ones(w, requires_grad=False).to(mat_W.device),
                )
                for _ in range(2):
                    u, v = (
                        torch.matmul(mat_W, v),
                        torch.matmul(mat_W.T, u),
                    )
                    v, u = (
                        v / torch.norm(v),
                        u / torch.norm(u),
                    )
                s = (u * torch.matmul(mat_W, v)).sum()
            loss = loss + s.norm()
        return self.lambda_ * loss


class OrthogonalNorm(Regularization):
    def forward(self, model):
        loss = 0
        for _, param in self._get_weight_list(model):
            mat_W = param
            if mat_W.dim() == 4:
                mat_W = torch.flatten(mat_W, start_dim=1)
            # # Here we list some different implementations of orthogonal regularization

            # # ||W @ W^T * (1 - I)||
            # mat_W = torch.matmul(mat_W, mat_W.t())
            # loss = loss + torch.sum(
            #     (
            #         mat_W
            #         * (
            #             torch.ones_like(mat_W)
            #             - torch.eye(mat_W.shape[0]).to(mat_W.device)
            #         )
            #     )
            #     ** 2
            # )

            # # ||W @ W^T - I||
            # mat_W = torch.matmul(mat_W, mat_W.t())
            # loss = loss + torch.norm(mat_W - torch.eye(mat_W.shape[0]).to(mat_W.device))

            # # ||W^T @ W - I||
            # mat_W = torch.matmul(mat_W.t(), mat_W)
            # loss = loss + torch.norm(mat_W - torch.eye(mat_W.shape[0]).to(mat_W.device))

            # ||normalize(W @ W^T) - I||
            mat_W = torch.matmul(mat_W, mat_W.t())
            loss = loss + torch.norm(
                mat_W / torch.norm(mat_W, dim=1, keepdim=True)
                - torch.eye(mat_W.shape[0]).to(mat_W.device)
            )
        return self.lambda_ * loss


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
        REG_ORTH,
        REG_SPEC,
    )
    from PGD import PGD
    from utils import DEVICE, Logger, get_train_test_dataloader

    # Hyperparameters
    num_epochs = NUM_EPOCHS
    learning_rate = LEARNING_RATE
    batch_size = BATCH_SIZE
    spec_lambda = REG_SPEC
    orth_lambda = REG_ORTH

    # Data
    train_loader, test_loader = get_train_test_dataloader(batch_size=batch_size)

    # Model
    model = CNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    spec_norm = SpectralNorm(lambda_=spec_lambda)
    orth_norm = OrthogonalNorm(lambda_=orth_lambda)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    logger = Logger("./log/Reg.log")

    if not os.path.exists("./model/Reg.ckpt"):
        # Train
        logger(
            f"Epochs: {num_epochs}, Batch size: {batch_size}, Learning rate: {learning_rate}, Spectral: {spec_lambda}, Orthogonal: {orth_lambda}"
        )
        loss_history = []
        model.train()
        for epoch in range(num_epochs):
            start = time()
            for i, (X, y) in enumerate(train_loader):
                X = X.to(DEVICE)
                y = y.to(DEVICE)
                optimizer.zero_grad()
                y_pred = model(X)
                reg_loss = spec_norm(model) + orth_norm(model)
                loss = criterion(y_pred, y) + reg_loss
                loss.backward()
                optimizer.step()
                if (i + 1) % 100 == 0:
                    loss_history.append(loss.item())
                    logger(
                        f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item()-reg_loss.item():.4f}+{reg_loss.item():.4f}"
                    )
            now = time()
            logger(f"Epoch time elapsed: {now - start:.3f} sec")
        # Save the model checkpoint
        torch.save(model.state_dict(), "./model/Reg.ckpt")
        pd.DataFrame(loss_history).to_csv("./log/Reg_loss_history.csv")
    else:
        # Load the model checkpoint
        model.load_state_dict(torch.load("./model/Reg.ckpt", map_location=DEVICE))

    # Testing
    total = 0
    correct = 0
    correct_adv = 0
    success = 0
    model.eval()
    attacker = PGD(eps=ATTACK_EPS, alpha=ATTACK_ALPHA, num_iter=ATTACK_ITER)

    for images, labels in tqdm(test_loader):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        total += labels.size(0)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
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
