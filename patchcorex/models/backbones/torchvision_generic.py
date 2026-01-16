import torch
import torchvision.models as models
from patchcorex.utils.registry import BACKBONES

@BACKBONES.register("torchvision")
@BACKBONES.register("resnet")
class TorchvisionBackbone:
    def __init__(self, model_name: str, pretrained: bool = True) -> None:
        if not hasattr(models, model_name):
            raise ValueError(f"Model {model_name} not found in torchvision.models")
        
        # Determine weights enum if pretrained
        weights = None
        if pretrained:
            # Try to find the default weights for the model
            model_name_capital = "".join([x.capitalize() for x in model_name.split("_")])
            # Handle some common cases where capitalization isn't simple (e.g. ResNet)
            if "Resnet" in model_name_capital:
                model_name_capital = model_name_capital.replace("Resnet", "ResNet")
            
            weight_attr = f"{model_name_capital}_Weights"
            if hasattr(models, weight_attr):
                weights_alias = getattr(models, weight_attr)
                weights = weights_alias.DEFAULT
        
        builder = getattr(models, model_name)
        self.model = builder(weights=weights)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def to(self, device):
        self.model.to(device)
        return self
