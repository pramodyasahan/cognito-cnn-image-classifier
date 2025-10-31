import torch.nn as nn
import torchvision.models as models

def get_resnet18(num_classes=10, pretrained=True):
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    in_features = m.fc.in_features
    m.fc = nn.Linear(in_features, num_classes)
    return m

def get_vgg16(num_classes=10, pretrained=True):
    m = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None)
    in_features = m.classifier[-1].in_features
    # replace last linear layer
    m.classifier[-1] = nn.Linear(in_features, num_classes)
    return m
