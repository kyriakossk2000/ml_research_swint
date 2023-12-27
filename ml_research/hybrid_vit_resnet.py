import torch
import torchvision.models as models


class HybridModel(torch.nn.Module):
    def __init__(self, resnet_backbone, swin_model, num_output_classes=10):
        super(HybridModel, self).__init__()
        self.resnet_stem = torch.nn.Sequential(
            resnet_backbone.conv1,
            resnet_backbone.bn1,
            resnet_backbone.relu,
            resnet_backbone.maxpool
        )
        self.resnet_bottlenecks = torch.nn.Sequential(*list(resnet_backbone.layer1.children())[:-1])
        self.swin = swin_model
        self.conv = torch.nn.Conv2d(64, 3, kernel_size=1, stride=1, padding=0) # add a 1x1 conv layer to reduce channels


    def forward(self, x):
        x = self.resnet_stem(x)
        x = self.resnet_bottlenecks(x)
        x = self.conv(x)
        x = self.swin(x)
        return x
