import torch, torchvision
from torchvision.models import resnet50,ResNet50_Weights
from torch import nn

class ResNet50Backbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = (ResNet50_Weights.DEFAULT if pretrained else None)
        model = resnet50(weights=weights)

        self.stem = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool
        )
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

    def forward(self,x):
        x = self.stem(x)

        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        return c2,c3,c4,c5


backbone = ResNet50Backbone(pretrained=True)

images = torch.randn(2, 3, 256, 704)

c2, c3, c4, c5 = backbone(images)

print("C2:", c2.shape)
print("C3:", c3.shape)
print("C4:", c4.shape)
print("C5:", c5.shape)