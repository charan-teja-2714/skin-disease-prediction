import timm
import torch.nn as nn

class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes=7, backbone="efficientnet_b3"):
        super().__init__()
        self.model = timm.create_model(
            backbone,
            pretrained=True,
            num_classes=num_classes
        )

    def freeze_backbone(self):
        """Freeze all layers except the classifier head."""
        for name, param in self.model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False

    def unfreeze_all(self):
        """Unfreeze all layers for full fine-tuning."""
        for param in self.model.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.model(x)
