import os

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

from CNN import CNN


class Quantize(autograd.Function):
    @staticmethod
    def forward(ctx, input, n_bits=8, a=None, b=None):
        # Get the max and min values from input
        # if a or b are not provided (not learnable)
        if a is None or b is None:
            a = input.min()
            b = input.max()
            ctx.save_for_backward(None, None, torch.ones_like(input).to(input.device))
        else:
            # The gradient for a and b are simple to the input bounds
            # This is equivalent to the EMA
            ctx.save_for_backward(
                a - input.min(),
                b - input.max(),
                torch.logical_and(input > a, input < b).to(input.device),
            )
        # Equally spaced quantization levels
        scale = (b - a) / (2 ** (n_bits - 1) - 1)
        return input.clamp(min=a, max=b).sub(a).div(scale).round().mul(scale).add(a)

    @staticmethod
    def backward(ctx, grad_output):
        grad_a, grad_b, mask = ctx.saved_tensors
        return grad_output * mask, None, grad_a, grad_b


class DTReLU(nn.Module):
    def __init__(self, n_bits=8):
        super().__init__()
        self.n_bits = n_bits
        self.lb = nn.parameter.Parameter(torch.zeros(1))
        self.ub = nn.parameter.Parameter(torch.ones(1))

    def forward(self, x):
        return Quantize.apply(F.relu(x), self.n_bits, self.lb, self.ub) * (x > 0)


class QWrap(nn.Module):
    def __init__(self, layer, n_bits=8):
        super().__init__()
        self.n_bits = n_bits
        self.layer = layer
        # We only wrap the convolutional layers and linear layers
        if isinstance(layer, nn.Linear):
            self.forward = self.__linear_forward
        elif isinstance(layer, nn.Conv2d):
            self.forward = self.__conv_forward
        else:
            raise NotImplementedError

    def __conv_forward(self, input):
        return F.conv2d(
            input,
            Quantize.apply(self.layer.weight, self.n_bits),
            self.layer.bias,
            self.layer.stride,
            self.layer.padding,
            self.layer.dilation,
            self.layer.groups,
        )

    def __linear_forward(self, input):
        return F.linear(
            input,
            Quantize.apply(self.layer.weight, self.n_bits),
            self.layer.bias,
        )


class QuantTrain(nn.Module):
    def __init__(
        self, model, w_bits=8, t_bits=8, quant_weights=True, quant_activation=True
    ) -> None:
        super().__init__()
        self.model = model
        for name, module in model.named_modules():
            # This is a bit hacky here, but things should work
            # as long as their structure are not too complicated
            if quant_weights and (
                isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear)
            ):
                model._modules[name.split(".")[0]][int(name.split(".")[1])] = QWrap(
                    module, w_bits
                )
            if quant_activation and isinstance(module, nn.ReLU):
                model._modules[name.split(".")[0]][int(name.split(".")[1])] = DTReLU(
                    t_bits
                )

    def forward(self, x):
        return self.model(x)


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
        QUANT_THRESH,
        QUANT_THRESH_BITS,
        QUANT_WEIGHT,
        QUANT_WEIGHT_BITS,
    )
    from PGD import PGD
    from utils import DEVICE, Logger, get_train_test_dataloader

    # Hyperparameters
    num_epochs = NUM_EPOCHS
    learning_rate = LEARNING_RATE
    batch_size = BATCH_SIZE
    enable_weight = QUANT_WEIGHT
    enable_thresh = QUANT_THRESH
    weight_bits = QUANT_WEIGHT_BITS
    thresh_bits = QUANT_THRESH_BITS

    # Data
    train_loader, test_loader = get_train_test_dataloader(batch_size=batch_size)

    # Model
    model = CNN().to(DEVICE)
    trainer = QuantTrain(
        model,
        w_bits=weight_bits,
        t_bits=thresh_bits,
        quant_weights=enable_weight,
        quant_activation=enable_thresh,
    ).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(trainer.parameters(), lr=learning_rate, momentum=0.9)
    logger = Logger("./log/Quant.log")

    if not os.path.exists("./model/Quant.ckpt"):
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
        torch.save(trainer.state_dict(), "./model/Quant.ckpt")
        pd.DataFrame(loss_history).to_csv("./log/Quant_loss_history.csv")
    else:
        # Load the model checkpoint
        trainer.load_state_dict(torch.load("./model/Quant.ckpt", map_location=DEVICE))

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
