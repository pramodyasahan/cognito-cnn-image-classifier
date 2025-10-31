import torch
import torch.nn as nn

class CustomCNN(nn.Module):
    def __init__(
        self,
        in_channels=1,
        num_classes=10,
        x1=32, m1=3,
        x2=64, m2=3,
        fc_units=128,
        dropout=0.3,
        activation="relu",
    ):
        super().__init__()
        act = nn.ReLU if activation.lower() == "relu" else nn.SiLU  # simple switch

        self.conv1 = nn.Conv2d(in_channels, x1, kernel_size=m1, padding=m1 // 2)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(x1, x2, kernel_size=m2, padding=m2 // 2)
        self.pool2 = nn.MaxPool2d(2, 2)

        # after two 2x2 pools on 28x28 -> 7x7 feature maps
        self.flatten_dim = x2 * 7 * 7

        self.fc = nn.Linear(self.flatten_dim, fc_units)
        self.drop = nn.Dropout(p=dropout)
        self.out = nn.Linear(fc_units, num_classes)

        self.act = act()

    def forward(self, x):
        x = self.pool1(self.act(self.conv1(x)))
        x = self.pool2(self.act(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.act(self.fc(x))
        x = self.drop(x)
        x = self.out(x)
        return x  # logits (we use CrossEntropyLoss which applies softmax internally)
