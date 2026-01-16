import torch
import torchvision.models as models
from patchcorex.utils.registry import BACKBONES

def build_wrn50_2(pretrained: bool = True):
    return models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1 if pretrained else None)

@BACKBONES.register("wide_resnet50_2")
@BACKBONES.register("wrn50_2")
class WideResNet50Backbone:
    def __init__(self, pretrained: bool = True) -> None:
        self.model = build_wrn50_2(pretrained=pretrained)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def to(self, device):
        self.model.to(device)
        return self