import torch, torchvision
from torch import nn
from ResNet50 import ResNet50Backbone
from torchvision.ops import FeaturePyramidNetwork

class MultiCameraBackbone(nn.Module):
    def __init__(self,pretrained):
        super().__init__()

        self.backbone = ResNet50Backbone(pretrained)
        self.fpn = FeaturePyramidNetwork(
            in_channels_list = [256, 512, 1024, 2048],
            out_channels=256
        )

    def forward(self,images):
        B,N,C,H,W = images.shape
        imgs = images.reshape(B*N,C,H,W)

        c2,c3,c4,c5 = self.backbone(imgs)
        features = {
            "0":c2,
            "1":c3,
            "2":c4,
            "3":c5
        }

        pyramid = self.fpn(features)
        outputs={}

        for key, value in pyramid.items():
            outputs[key] = value.reshape(
                B,N,value.shape[1],value.shape[2],value.shape[3]
            )
        return outputs

#Test
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

model = MultiCameraBackbone(
    pretrained=True
).to(device)

images = torch.randn(
    2, 6, 3, 128, 352
).to(device)

features = model(images)

for key, value in features.items():

    print(key, value.shape)