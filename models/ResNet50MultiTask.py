from torchvision import models
import torch.nn as nn
import torch

class ClinicalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.fc(x)

class EMInteraction(nn.Module):
    def __init__(self, feat_dim, num_tasks=2, num_bases=8, iters=3, tau=0.5):
        super().__init__()
        self.num_bases = num_bases
        self.iters = iters
        self.tau = tau
        self.mu = nn.Parameter(torch.randn(num_bases, feat_dim) * 0.01)

    def forward(self, features):  # (B, T, C)
        B, T, C = features.shape
        X = features.view(B * T, C)
        mu = self.mu

        for _ in range(self.iters):
            attn = torch.softmax(X @ mu.T / self.tau, dim=-1)
            mu = (attn.T @ X) / (attn.sum(dim=0, keepdim=True).T + 1e-6)

        X_tilde = attn @ mu
        return X_tilde.view(B, T, C)

class SimpleResNet50MultiTask(nn.Module):
    def __init__(self, clinical_dim=4, fusion_dim=512):
        super().__init__()
        base = models.resnet50(weights=True)

        original_conv = base.conv1
        new_conv = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            new_conv.weight[:, :3] = original_conv.weight
            new_conv.weight[:, 3:] = original_conv.weight
        base.conv1 = new_conv
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        self.image_fc = nn.Linear(base.fc.in_features, fusion_dim)
        self.clinical_fc = ClinicalEncoder(clinical_dim, output_dim=fusion_dim)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.ReLU()
        )

        self.task_proj = nn.ModuleList([nn.Linear(fusion_dim, fusion_dim) for _ in range(3)])
        self.em_stage1 = EMInteraction(feat_dim=fusion_dim, num_tasks=2)
        self.em_stage2 = EMInteraction(feat_dim=fusion_dim, num_tasks=2)

        self.cclnm_head = nn.Sequential(nn.Linear(fusion_dim, 64), nn.ReLU(), nn.Linear(64, 1))
        self.lclnm_head = nn.Sequential(nn.Linear(fusion_dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, image, clinical):

        x_img = self.backbone(image).view(image.size(0), -1)
        x_img = self.image_fc(x_img)
        x_clin = self.clinical_fc(clinical)
        shared = self.fusion(torch.cat([x_img, x_clin], dim=-1))  # (B, C)


        tasks_stage1 = torch.stack([proj(shared) for proj in self.task_proj[:2]], dim=1)  # (B, 2, C)
        inter1 = self.em_stage1(tasks_stage1) + tasks_stage1


        task1_feat = inter1[:, 0]
        task2_feat = inter1[:, 1]

        tasks_stage2 = torch.stack([
            self.task_proj[0](task1_feat),
            self.task_proj[1](task2_feat)
        ], dim=1)
        inter2 = self.em_stage2(tasks_stage2) + tasks_stage2


        cclnm = torch.sigmoid(self.cclnm_head(inter2[:, 0]))
        lclnm = torch.sigmoid(self.lclnm_head(inter2[:, 1]))

        return {
            'cclnm': cclnm,
            'lclnm': lclnm,
            'cclnm_feat': inter2[:, 0],
            'lclnm_feat': inter2[:, 1]
        }