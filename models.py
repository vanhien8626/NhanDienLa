import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


class Classifier(nn.Module):
    def __init__(self, classes, config, architecture='cnn', pretrained=True):
        super().__init__()
        self.architecture = architecture
        self._frozen = []
        if architecture == 'cnn':
            channels = [3, 16, 32, 64]
            self.blocks = nn.ModuleList([
                nn.Sequential(nn.Conv2d(channels[i], channels[i + 1], 3, padding=1),
                              nn.BatchNorm2d(channels[i + 1]), nn.ReLU(), nn.MaxPool2d(2))
                for i in range(3)])
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.head = nn.Sequential(nn.Dropout(config.dropout), nn.Linear(64, classes))
            groups = list(self.blocks)
        elif architecture == 'resnet18':
            self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
            self.backbone.fc = nn.Sequential(nn.Dropout(config.dropout), nn.Linear(512, classes))
            groups = [nn.Sequential(self.backbone.conv1, self.backbone.bn1),
                      self.backbone.layer1, self.backbone.layer2, self.backbone.layer3]
        else:
            raise ValueError('Architecture must be cnn or resnet18.')
        for group in groups[:config.frozen_blocks]:
            for parameter in group.parameters():
                parameter.requires_grad = False
            self._frozen.append(group)

    def train(self, mode=True):
        super().train(mode)
        for module in self._frozen:
            module.eval()  # Frozen BatchNorm must not keep updating its statistics.
        return self

    @property
    def target_layer(self):
        return self.blocks[-1][2] if self.architecture == 'cnn' else self.backbone.layer4

    def forward(self, inputs):
        if self.architecture == 'resnet18':
            return self.backbone(inputs)
        for block in self.blocks:
            inputs = block(inputs)
        return self.head(self.pool(inputs).flatten(1))


def parameter_counts(model):
    return {'total': sum(p.numel() for p in model.parameters()),
            'trainable': sum(p.numel() for p in model.parameters() if p.requires_grad)}
