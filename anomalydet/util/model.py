import torch.nn as nn
import timm


class TimmAlignClassifier(nn.Module):
    def __init__(self, backbone: str, pretrained: bool, embed_dim: int, num_class: int):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        feat_dim = self.backbone.num_features
        self.embed = nn.Linear(feat_dim, embed_dim)
        self.cls = nn.Linear(embed_dim, num_class)

    def forward(self, x):
        feat = self.backbone(x)
        emb = self.embed(feat)
        logits = self.cls(emb)
        return emb, logits
