import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from utils.MultiTaskDataset import GastricDataset
from sklearn.metrics import roc_auc_score, accuracy_score
from lifelines.utils import concordance_index
import numpy as np
from models.ResNet50MultiTask import SimpleResNet50MultiTask

LAMBDA_LLNM = 1.0
LEARNING_RATE = 1e-5
LR_DECAY_STEP = 50
LR_DECAY_GAMMA = 0.5
MAX_EPOCHS = 200
EARLY_STOP_PATIENCE = 20


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    y_true_cclnm, y_pred_cclnm = [], []
    y_true_lclnm, y_pred_lclnm = [], []

    for image, clinical, cclnm, lclnm, ids in dataloader:
        image, clinical = image.to(device), clinical.to(device)
        cclnm, lclnm = cclnm.to(device), lclnm.to(device)

        output = model(image, clinical)
        pred_cclnm, pred_lclnm = output['cclnm'], output['lclnm']

        y_true_cclnm.append(cclnm.cpu().numpy())
        y_pred_cclnm.append(pred_cclnm.cpu().numpy())
        y_true_lclnm.append(lclnm.cpu().numpy())
        y_pred_lclnm.append(pred_lclnm.cpu().numpy())

    y_true_cclnm = np.concatenate(y_true_cclnm)
    y_pred_cclnm = np.concatenate(y_pred_cclnm)
    y_true_lclnm = np.concatenate(y_true_lclnm)
    y_pred_lclnm = np.concatenate(y_pred_lclnm)

    auc_cclnm = roc_auc_score(y_true_cclnm, y_pred_cclnm)
    acc_cclnm = accuracy_score(y_true_cclnm > 0.5, y_pred_cclnm > 0.5)
    auc_lclnm = roc_auc_score(y_true_lclnm, y_pred_lclnm)
    acc_lclnm = accuracy_score(y_true_lclnm > 0.46, y_pred_lclnm > 0.46)

    return auc_cclnm, acc_cclnm, auc_lclnm, acc_lclnm


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleResNet50MultiTask().to(device)
    if os.path.exists('Results/weights'):
        model.load_state_dict(torch.load('Results/weights', map_location=device))
        print("Resumed from existing checkpoint: Results/weights")

    transform1 = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize([0.18461268, 0.18459961, 0.1887984],
                             [0.19145657, 0.19151175, 0.19579896])
    ])
    transform2 = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.00038843, 0.00038843, 0.00038843],
                             [0.00617307, 0.00617307, 0.00617307])
    ])

    transform3 = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.182533, 0.18200818, 0.1867373],
                             [0.19620422, 0.1956934, 0.20086534])
    ])
    transform4 = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.00030449, 0.00030449, 0.00030449],
                             [0.00517094, 0.00517094, 0.00517094])
    ])

    train_set = GastricDataset('data/train.csv', 'data/PTC/tumor/train', 'data/PTC/fat/train', transform1, transform2)
    val_set = GastricDataset('data/PTC.csv', 'data/PTC/tumor/valid', 'data/PTC/fat/valid', transform3, transform4)

    cclnm_labels = train_set.df['Label1'].values.astype(int)
    lclnm_labels = train_set.df['Label2'].values.astype(int)

    combined_labels = [f"{c}_{l}" for c, l in zip(cclnm_labels, lclnm_labels)]

    from collections import Counter

    label_counts = Counter(combined_labels)

    total = len(combined_labels)

    weights = [total / label_counts[label] for label in combined_labels]

    sampler = WeightedRandomSampler(weights, num_samples=total, replacement=True)

    train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=8)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=LR_DECAY_STEP, gamma=LR_DECAY_GAMMA)
    bce = nn.BCELoss()

    best_val_auc = 0.0
    epochs_no_improve = 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        total_loss = 0
        for image, clinical, cclnm, lclnm, ids in train_loader:
            image, clinical = image.to(device), clinical.to(device)
            cclnm, lclnm = cclnm.to(device), lclnm.to(device)

            output = model(image, clinical)
            pred_cclnm, pred_lclnm = output['cclnm'], output['lclnm']

            loss_cclnm = bce(pred_cclnm.squeeze(dim=1), cclnm)
            loss_lclnm = bce(pred_lclnm.squeeze(dim=1), lclnm)
            loss = loss_cclnm + LAMBDA_LLNM * loss_lclnm

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        auc_cclnm, acc_cclnm, auc_lclnm, acc_lclnm = evaluate(model, val_loader, device)
        val_auc = (auc_cclnm + auc_lclnm) / 2
        print(
            f"Epoch {epoch + 1}, Loss: {total_loss:.3f}, LR: {scheduler.get_last_lr()[0]:.2e}, "
            f"AUC_CCLNM: {auc_cclnm:.3f}, ACC_CCLNM: {acc_cclnm:.3f}, "
            f"AUC_LCLNM: {auc_lclnm:.3f}, ACC_LCLNM: {acc_lclnm:.3f}"
        )

        os.makedirs('Results', exist_ok=True)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'Results/weights')
            print(f"  -> New best mean AUC: {best_val_auc:.3f}, checkpoint saved.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                print(f"Early stopping triggered at epoch {epoch + 1} "
                      f"(no improvement for {EARLY_STOP_PATIENCE} epochs).")
                break