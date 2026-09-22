import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class QualityModel(nn.Module):
    """Five-class image quality model. Outputs a quality distribution over Q_LEVELS."""

    def __init__(self, model_name='efficientnet', num_classes=5, device='cuda:0'):
        super().__init__()
        if model_name == 'efficientnet':
            backbone = models.efficientnet_b2(weights='IMAGENET1K_V1')
            self.features = nn.Sequential(*list(backbone.children())[:-1])
            in_features = backbone.features[-1].out_channels
        elif model_name == 'resnet':
            backbone = models.resnet34(weights='IMAGENET1K_V1')
            self.features = nn.Sequential(*list(backbone.children())[:9])
            in_features = 512
        elif model_name == 'densenet':
            backbone = models.densenet121(weights='IMAGENET1K_V1')
            self.features = nn.Sequential(*list(backbone.features.children()), nn.AdaptiveAvgPool2d((1, 1)))
            in_features = 1024
        else:
            raise ValueError(f'unknown model_name: {model_name}')

        self.linear = nn.Linear(in_features, num_classes)
        self.to(device)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        return F.gumbel_softmax(x, tau=1, hard=False, dim=-1)
