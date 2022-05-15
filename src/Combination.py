import os

import torch
import torch.nn as nn

from Adversarial import AdvTrain
from PGD import PGD
from Quantization import QuantTrain


class CombineTrain(nn.Module):
    def __init__(
        self,
        init_model,
        adversarial=True,
        adv_eps=8 / 255,
        adv_alpha=2 / 255,
        adv_iter=20,
        quant_weight=True,
        quant_thresh=True,
        w_bits=8,
        t_bits=8,
    ) -> None:
        super().__init__()
        self.model = init_model
        if quant_weight or quant_thresh:
            self.model = QuantTrain(
                self.model, w_bits, t_bits, quant_weight, quant_thresh
            )
        if adversarial:
            self.model = AdvTrain(
                self.model,
                PGD(eps=adv_eps, alpha=adv_alpha, num_iter=adv_iter),
            )

    def forward(self, X, *args):
        return self.model(X, *args)


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
        REG_ORTH,
        REG_SPEC,
    )
    from Regularization import OrthogonalNorm, SpectralNorm
    from utils import DEVICE, Logger, get_train_test_dataloader

    # Hyper-hyperparameters
    # Adv, Quant, Reg
    hhps = [
        (True, True, True),
        (False, True, True),
        (True, False, True),
        (True, True, False),
    ]

    # Hyperparameters
    for adversarial, quantization, regularization in hhps:
        attack_eps = ATTACK_EPS
        attack_alpha = ATTACK_ALPHA
        attack_iter = ATTACK_ITER
        enable_weight = QUANT_WEIGHT
        enable_thresh = QUANT_THRESH
        weight_bits = QUANT_WEIGHT_BITS
        thresh_bits = QUANT_THRESH_BITS
        num_epochs = NUM_EPOCHS
        learning_rate = LEARNING_RATE
        batch_size = BATCH_SIZE
        spec_lambda = REG_SPEC
        orth_lambda = REG_ORTH
        strategy_name = (
            "Comb_"
            + ("Quant_" if quantization else "")
            + ("Adv_" if adversarial else "")
            + ("Reg_" if regularization else "")
            + "Train"
        )

        # Data
        train_loader, test_loader = get_train_test_dataloader(batch_size=batch_size)

        # Model
        base_model = CNN().to(DEVICE)
        model = CombineTrain(
            base_model,
            adversarial=adversarial,
            adv_eps=attack_eps,
            adv_alpha=attack_alpha,
            adv_iter=attack_iter,
            quant_thresh=quantization and enable_thresh,
            quant_weight=quantization and enable_weight,
            w_bits=weight_bits,
            t_bits=thresh_bits,
        ).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        spec_norm = SpectralNorm(lambda_=spec_lambda)
        orth_norm = OrthogonalNorm(lambda_=orth_lambda)
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        logger = Logger(f"./log/{strategy_name}.log")

        if not os.path.exists(f"./model/{strategy_name}.ckpt"):
            # Train
            logger(
                f"Epochs: {num_epochs}, Batch size: {batch_size}, Learning rate: {learning_rate}"
                + (
                    f", Spectral: {spec_lambda}, Orthogonal: {orth_lambda}"
                    if regularization
                    else ""
                )
                + (f", Quantize weights: {weight_bits} bits" if enable_weight else "")
                + (
                    f", Quantize thresholds: {thresh_bits} bits"
                    if enable_thresh
                    else ""
                )
            )
            loss_history = []
            model.train()
            for epoch in range(num_epochs):
                start = time()
                for i, (X, y) in enumerate(train_loader):
                    X = X.to(DEVICE)
                    y = y.to(DEVICE)
                    optimizer.zero_grad()
                    if adversarial:
                        y_pred = model(X, y)
                    else:
                        y_pred = model(X)
                    loss = criterion(y_pred, y)
                    if regularization:
                        reg_loss = orth_norm(model) + spec_norm(model)
                        loss += reg_loss
                    loss.backward()
                    optimizer.step()
                    if (i + 1) % 100 == 0:
                        loss_history.append(loss.item())
                        logger(
                            f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: "
                            + (
                                f"{loss.item()-reg_loss.item():.4f}+{reg_loss.item():.4f}"
                                if regularization
                                else f"{loss.item():.4f}"
                            )
                        )
                now = time()
                logger(f"Epoch time elapsed: {now - start:.3f} sec")
            # Save the model checkpoint
            torch.save(model.state_dict(), f"./model/{strategy_name}.ckpt")
            pd.DataFrame(loss_history).to_csv(f"./log/{strategy_name}_loss_history.csv")
        else:
            # Load the model checkpoint
            model.load_state_dict(
                torch.load(f"./model/{strategy_name}.ckpt", map_location=DEVICE)
            )

        # Test
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
            success += -(
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
