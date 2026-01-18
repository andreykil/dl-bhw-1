import numpy as np
from tqdm import tqdm
import torch

def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    mixup_alpha=0.0,
):
    model.train()

    running_loss = 0.0
    correct = 0.0
    total = 0

    for images, labels in tqdm(dataloader, desc="Train", leave=False):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()


        if mixup_alpha > 0:
            # Mixup аугментация
            lam = np.random.beta(mixup_alpha, mixup_alpha)
            index = torch.randperm(images.size(0)).to(device)
            mixed_images = lam * images + (1 - lam) * images[index]
            labels_a, labels_b = labels, labels[index]
            outputs = model(mixed_images)
            loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
        else:
            # Стандартный forward без Mixup
            outputs = model(images)
            loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        preds = outputs.argmax(dim=1)
        
        if mixup_alpha > 0:
            # Взвешенная accuracy для Mixup
            correct += lam * (preds == labels_a).sum().item() + (1 - lam) * (preds == labels_b).sum().item()
        else:
            correct += (preds == labels).sum().item()

        total += labels.size(0)

    avg_loss = running_loss / total
    accuracy = correct / total

    return avg_loss, accuracy
