from backbone import backbone
from decoder import Decoder
import torch
import torch.nn as nn


#model_url = './Cityscapes_best.pth.tar'


class DFANet2(nn.Module):

    def __init__(self, n_classes=19, pretrained=False, pretrained_backbone=True):
        super(DFANet2, self).__init__()
        self.backbone1 = backbone(pretrained=pretrained_backbone)
        self.backbone1_up = nn.UpsamplingBilinear2d(scale_factor=4)

        self.backbone2 = backbone(pretrained=pretrained_backbone)
        self.backbone2_up = nn.UpsamplingBilinear2d(scale_factor=4)

        self.backbone3 = backbone(pretrained=pretrained_backbone)

        self.decoder = Decoder(n_classes=n_classes)

        # if pretrained:
        #     self.load_state_dict(torch.load(model_url)["state_dict"])

    def forward(self, x):
        enc1_2, enc1_3, enc1_4, fc1, fca1 = self.backbone1(x)
        fca1_up = self.backbone1_up(fca1)

        enc2_2, enc2_3, enc2_4, fc2, fca2 = self.backbone2.forward_concat(fca1_up, enc1_2, enc1_3, enc1_4)
        fca2_up = self.backbone2_up(fca2)

        enc3_2, enc3_3, enc3_4, fc3, fca3 = self.backbone3.forward_concat(fca2_up, enc2_2, enc2_3, enc2_4)

        out = self.decoder(enc1_2, enc2_2, enc3_2, fca1, fca2, fca3)

        return out