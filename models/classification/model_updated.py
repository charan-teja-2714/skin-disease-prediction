import torch
import torch.nn as nn
from torchvision import models

class EfficientNetV2LFeatureExtractor(nn.Module):
    def __init__(self):
        super(EfficientNetV2LFeatureExtractor, self).__init__()
        # Load EfficientNetV2-L with the best available ImageNet weights
        # This model is significantly more powerful than standard EfficientNet
        weights = models.EfficientNet_V2_L_Weights.IMAGENET1K_V1
        self.base_model = models.efficientnet_v2_l(weights=weights)
        
        # Feature extraction backbone (all layers except classifier)
        self.features = self.base_model.features
        self.avgpool = self.base_model.avgpool
        
        # EfficientNetV2-L output features size is typically 1280
        self.output_dim = 1280 

    def forward(self, x):
        # Pass through the feature extractor
        x = self.features(x)
        # Pool spatial dimensions (7x7 -> 1x1)
        x = self.avgpool(x)
        # Flatten to (Batch_Size, 1280)
        x = torch.flatten(x, 1)
        return x