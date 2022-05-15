import torch
import torch.nn as nn


class PGD:
    def __init__(
        self, eps=10 / 255, alpha=2 / 255, num_iter=30, rand_init=True
    ) -> None:
        self.eps = eps
        self.alpha = alpha
        self.num_iter = num_iter
        self.rand_init = rand_init
        self.criterion = nn.CrossEntropyLoss()

    def attack(self, model, X, y):
        X_adv = X.clone()
        if self.rand_init:
            # This is normally recommended for PGD attack,
            # but doesn't do that a lot harm if discarded
            X_adv = X_adv + torch.normal(mean=0, std=self.eps, size=X_adv.shape).to(
                X.device
            )
        for _ in range(self.num_iter):
            X_adv.requires_grad = True
            with torch.enable_grad():
                output = model(X_adv)
                loss = self.criterion(output, y)
                loss.backward()
            # Update image and clear gradient
            X_adv = X_adv.detach() + self.alpha * torch.sign(X_adv.grad)
            X_adv = torch.clamp(X_adv, X - self.eps, X + self.eps)
            X_adv = torch.clamp(X_adv, -2, 2)
        X_adv.requires_grad = False
        return X_adv


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    import numpy as np

    from CNN import CNN
    from params import ATTACK_ALPHA, ATTACK_EPS, ATTACK_ITER
    from utils import DEVICE, class_labels, get_test_dataloader

    # Model & Data
    model = CNN().to(DEVICE)
    model.load_state_dict(torch.load("./model/CNN.ckpt", map_location=DEVICE))
    model.eval()
    attacker = PGD(eps=ATTACK_EPS, alpha=ATTACK_ALPHA, num_iter=ATTACK_ITER)
    test_loader = get_test_dataloader(batch_size=100)

    # Evaluation of model and attacker
    total = 0
    correct = 0
    correct_adv = 0
    success = 0
    success_histories = []
    for images, labels in tqdm(test_loader):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        total += labels.size(0)
        # Model evaluation
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        # Attack
        adv_images = attacker.attack(model, images, labels)
        outputs = model(adv_images)
        _, predicted_adv = torch.max(outputs.data, 1)
        correct_adv += (predicted_adv == labels).sum().item()
        success -= (
            torch.where(predicted == labels, predicted_adv == labels, False)
            .sum()
            .item()
        )

        for i in range(images.size(0)):
            # Record successful attacks
            if (predicted[i] == labels[i]) and (predicted_adv[i] != labels[i]):
                success_histories.append(
                    (
                        images[i].detach().cpu().numpy(),
                        adv_images[i].detach().cpu().numpy(),
                        labels[i].detach().cpu().numpy(),
                        predicted[i].detach().cpu().numpy(),
                        predicted_adv[i].detach().cpu().numpy(),
                    )
                )
    success += correct

    # Visualization
    vis_size = 5
    success_histories = [
        success_histories[i]
        for i in np.random.choice(len(success_histories), vis_size, replace=False)
    ]

    fig = plt.figure(figsize=(12, 6))
    plt.tight_layout()
    plt.title(
        f"Model accuracy : {100 * correct / total:.2f}% -> {100 * correct_adv / total:.2f}%, "
        f"Success rate : {100 * success / correct:.2f}% "
        f"($\\epsilon={int(ATTACK_EPS *255)} / 255$)"
    )
    plt.axis("off")
    for i, (image, image_adv, label, pred, pred_adv) in enumerate(success_histories):
        ax = fig.add_subplot(2, vis_size, i + 1)
        # Original image
        ax.imshow(image.squeeze().transpose(1, 2, 0))
        ax.set_title(f"{class_labels[int(label)]}")
        ax.set_xlabel(f"{class_labels[int(pred)]}")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        # Adversarial image
        ax = fig.add_subplot(2, vis_size, i + 1 + vis_size)
        ax.imshow(image_adv.squeeze().transpose(1, 2, 0))
        ax.set_xlabel(f"{class_labels[int(pred_adv)]}")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
    fig.savefig("./fig/PGD.pdf")
    # plt.show()
