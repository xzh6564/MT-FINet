import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from torch.utils.data import DataLoader
from models.ResNet50MultiTask import SimpleResNet50MultiTask
from utils.MultiTaskDataset import GastricDataset
from torchvision import transforms
import pandas as pd

def compute_confidence_interval(metric_fn, y_true, y_pred, n_bootstraps=100, alpha=0.95):
    rng = np.random.RandomState(42)
    scores = []
    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[indices])) < 2:
            continue
        score = metric_fn(y_true[indices], y_pred[indices])
        scores.append(score)
    sorted_scores = np.sort(scores)
    lower = np.percentile(sorted_scores, ((1.0 - alpha) / 2.0) * 100)
    upper = np.percentile(sorted_scores, (alpha + ((1.0 - alpha) / 2.0)) * 100)
    return lower, upper

@torch.no_grad()
def predict(model, dataloader, device):
    model.eval()

    ids_all = []
    cclnm_feats_all, lclnm_feats_all = [], []
    cclnm_probs_all, cclnm_preds_all, cclnm_labels_all = [], [], []
    lclnm_probs_all, lclnm_preds_all, lclnm_labels_all = [], [], []

    for batch in dataloader:
        image, clinical, cclnm, lclnm, ids = batch
        image, clinical = image.to(device), clinical.to(device)
        output = model(image, clinical)
        pred_cclnm, pred_lclnm = output['cclnm'], output['lclnm']
        feat_cclnm, feat_lclnm = output['cclnm_feat'], output['lclnm_feat']

        cclnm_feats_all.append(feat_cclnm.cpu().numpy())
        lclnm_feats_all.append(feat_lclnm.cpu().numpy())
        # CCLNM
        cclnm_probs = pred_cclnm.flatten().cpu().numpy()
        cclnm_preds = (cclnm_probs > 0.02).astype(int)
        cclnm_labels = cclnm.cpu().numpy().reshape(-1)

        # LCLNM
        lclnm_probs = pred_lclnm.flatten().cpu().numpy()
        lclnm_preds = (lclnm_probs > 0.20).astype(int)
        lclnm_labels = lclnm.cpu().numpy().reshape(-1)

        cclnm_probs_all.extend(cclnm_probs)
        cclnm_preds_all.extend(cclnm_preds)
        cclnm_labels_all.extend(cclnm_labels)

        lclnm_probs_all.extend(lclnm_probs)
        lclnm_preds_all.extend(lclnm_preds)
        lclnm_labels_all.extend(lclnm_labels)

        ids_all.extend(ids)

    return (
        np.array(ids_all),
        np.array(cclnm_labels_all), np.array(cclnm_probs_all), np.array(cclnm_preds_all),
        np.array(lclnm_labels_all), np.array(lclnm_probs_all), np.array(lclnm_preds_all),
        np.concatenate(cclnm_feats_all, axis=0),
        np.concatenate(lclnm_feats_all, axis=0)
    )

def evaluate_classification(y_true, y_prob, y_pred, name):
    auc = roc_auc_score(y_true, y_prob)
    acc = accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sens = tp / (tp + fn + 1e-6)
    spec = tn / (tn + fp + 1e-6)

    auc_ci = compute_confidence_interval(roc_auc_score, y_true, y_prob)
    acc_ci = compute_confidence_interval(accuracy_score, y_true, y_pred)
    sens_ci = compute_confidence_interval(lambda y, p: confusion_matrix(y, p).ravel()[3] / (confusion_matrix(y, p).ravel()[3] + confusion_matrix(y, p).ravel()[2] + 1e-6), y_true, y_pred)
    spec_ci = compute_confidence_interval(lambda y, p: confusion_matrix(y, p).ravel()[0] / (confusion_matrix(y, p).ravel()[0] + confusion_matrix(y, p).ravel()[1] + 1e-6), y_true, y_pred)

    print(f"AUC_{name}: {auc:.3f} [{auc_ci[0]:.3f}, {auc_ci[1]:.3f}]")
    print(f"ACC_{name}: {acc:.3f} [{acc_ci[0]:.3f}, {acc_ci[1]:.3f}]")
    print(f"SENS_{name}: {sens:.3f} [{sens_ci[0]:.3f}, {sens_ci[1]:.3f}]")
    print(f"SPEC_{name}: {spec:.3f} [{spec_ci[0]:.3f}, {spec_ci[1]:.3f}]")

    return {
        f'True_Label_{name}': y_true,
        f'Pred_Label_{name}': y_pred,
        f'Prob_{name}': y_prob,
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleResNet50MultiTask().to(device)
    model.load_state_dict(torch.load("./Results/weights", map_location=device))


    model.eval()

    transform1 = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.182533, 0.18200818, 0.1867373],
                             [0.19620422, 0.1956934, 0.20086534])
    ])

    transform2 = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.00030449, 0.00030449, 0.00030449],
                             [0.00517094, 0.00517094, 0.00517094])
    ])

    val_dataset = GastricDataset('data/PTC.csv', 'data/PTC/tumor/valid', 'data/PTC/fat/valid', transform1, transform2)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, drop_last=True)

    (
        ids,
        y_true_cclnm, y_prob_cclnm, y_pred_cclnm,
        y_true_lclnm, y_prob_lclnm, y_pred_lclnm,
        cclnm_feats, lclnm_feats
    ) = predict(model, val_loader, device)

    cclnm_result = evaluate_classification(y_true_cclnm, y_prob_cclnm, y_pred_cclnm, "CCLNM")
    lclnm_result = evaluate_classification(y_true_lclnm, y_prob_lclnm, y_pred_lclnm, "LCLNM")

    df = pd.DataFrame({
        "ID": ids,
        "y_true_cclnm": y_true_cclnm,
        "y_pred_cclnm": y_pred_cclnm,
        "y_prob_cclnm": y_prob_cclnm,
        "y_true_lclnm": y_true_lclnm,
        "y_pred_lclnm": y_pred_lclnm,
        "y_prob_lclnm": y_prob_lclnm,
    })


if __name__ == "__main__":
    main()
