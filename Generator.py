import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchvision.models.resnet import ResNet, Bottleneck
import config


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.bn1(self.relu(self.conv1(x)))
        x = self.bn2(self.relu(self.conv2(x)))
        return x



class UNetResNet18(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetResNet18, self).__init__()

        self.encoder = models.resnet18(pretrained=True)
        self.encoder.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.encoder.fc = nn.Identity()

        self.decoder = nn.ModuleList([
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ConvTranspose2d(512, 128, kernel_size=2, stride=2),
            nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2),
            nn.ConvTranspose2d(128, 32, kernel_size=2, stride=2),
            nn.ConvTranspose2d(96, 32, kernel_size=2, stride=2),
            nn.Conv2d(32, out_channels, kernel_size=1)
        ])

        self.bottle_neck = DoubleConv(512, 512)

        # Add dropout layers
        self.dropout = nn.Dropout2d(p=0.1)

    def forward(self, x):
        # Encoder
        x1 = self.encoder.conv1(x)
        x2 = self.encoder.layer1(self.encoder.maxpool(F.relu(x1)))
        x3 = self.encoder.layer2(x2)
        x4 = self.encoder.layer3(x3)
        x5 = self.encoder.layer4(x4)

        Bottleneck = self.bottle_neck(x5)
        # Bottleneck = self.dropout(Bottleneck)

        # Decoder
        x = self.decoder[0](Bottleneck)
        x = torch.cat([x, x4], dim=1)
        x = self.decoder[1](x)
        x = torch.cat([x, x3], dim=1)
        # x = self.dropout(x)
        x = self.decoder[2](x)
        x = torch.cat([x, x2], dim=1)
        # x = self.dropout(x)
        x = self.decoder[3](x)
        x = torch.cat([x, x1], dim=1)
        # x = self.dropout(x)
        x = self.decoder[4](x)
        x = self.decoder[5](x)

        return x
    

def test():
    x = torch.randn((1, 1, 256, 256))
    model_resnet = UNetResNet18(1,2)
    preds_resnet = model_resnet(x)
    print(preds_resnet.shape)


if __name__ == "__main__":
    test()
