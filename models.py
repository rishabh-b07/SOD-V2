import torch
from torch import nn
from torchvision import models

class EfficientNetBackbone(nn.Module):
  def __init__(self, variant="b0"):
    super().__init__()
    self.model = models.efficientnet.(variant)(pretrained=True)
    for param in self.model.parameters():
      param.requires_grad = False

  def forward(self, x):
    x = self.model.extract_features(x)
    return x

class CBAM(nn.Module):
  def __init__(self, channel_in):
    super().__init__()
    self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
    self.channel_attention = nn.Sequential(
        nn.Linear(channel_in, channel_in // 2),
        nn.ReLU(inplace=True),
        nn.Linear(channel_in // 2, channel_in),
        nn.Sigmoid()
    )
    self.max_pool = nn.AdaptiveMaxPool2d(output_size=(1, 1))
    self.spatial_attention = nn.Sequential(
        nn.Conv2d(channel_in, channel_in, kernel_size=7, padding=3, bias=False),
        nn.ReLU(inplace=True),
        nn.Conv2d(channel_in, 1, kernel_size=7, padding=3, bias=False),
        nn.Sigmoid()
    )

  def forward(self, x):
    avg_out = self.avg_pool(x)  
    channel_att = self.channel_attention(avg_out)
    channel_att = channel_att.unsqueeze(2).unsqueeze(3)  

    max_out = self.max_pool(x) 
    spatial_att = self.spatial_attention(max_out)

    out = x * channel_att + x * spatial_att
    return out

class CustomYOLOv9(nn.Module):
  def __init__(self, num_classes, backbone_variant="b0"):
    super().__init__()
    self.backbone = EfficientNetBackbone(backbone_variant)
    self.head = nn.Sequential(
        nn.Conv2d(in_channels=..., out_channels=..., kernel_size=..., padding=...),
         nn.BatchNorm2d(num_features=...),
         nn.LeakyReLU(negative_slope=0.1),
    )

  def forward(self, x):
    features = self.backbone(x)
    predictions = self.head(features)
    return predictions

model = CustomYOLOv9(num_classes=80)  
