import torch
import torch.nn as nn
import timm


class TimmOstClassifier(nn.Module):
    def __init__(self, backbone: str, pretrained: bool, embed_dim: int, num_class: int):
        super().__init__()
        # backbone outputs feature if num_classes=0
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        in_dim = self.backbone.num_features

        self.embed = nn.Linear(in_dim, embed_dim)
        self.bn = nn.BatchNorm1d(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_class)

    def forward(self, x):
        feat = self.backbone(x)               # (B, in_dim)
        emb = self.embed(feat)                # (B, embed_dim)
        emb = self.bn(emb)
        logits = self.classifier(emb)         # (B, C)
        return emb, logits
