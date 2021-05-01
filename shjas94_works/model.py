import torch
import torch.nn as nn
from torch.nn import functional as F
from efficientnet_pytorch import EfficientNet


class EffNet(nn.Module):
    def __init__(self):
        super(EffNet, self).__init__()
        effnet = EfficientNet.from_pretrained('efficientnet-b7')
        head = nn.Sequential(effnet._conv_stem, effnet._bn0)
        blocks = list(effnet._blocks.children())
        tail = nn.Sequential(effnet._conv_head, effnet._bn1)
        blocks.insert(0, head)
        # blocks.append(nn.Dropout(0.7, inplace=False))
        blocks.append(tail)
        blocks.append(nn.Conv2d(2560, 512, 1, bias=False))  # projection
        self.backbone = nn.Sequential(*blocks)

    def forward(self, x):
        output = self.backbone(x)
        return output


class EffNet2(nn.Module):
    def __init__(self):
        super(EffNet2, self).__init__()
        effnet = EfficientNet.from_pretrained('efficientnet-b7')
        head = nn.Sequential(effnet._conv_stem, effnet._bn0)
        blocks_head = list(effnet._blocks.children())[:11]
        blocks_tail = list(effnet._blocks.children())[11:]
        tail = nn.Sequential(effnet._conv_head, effnet._bn1)
        blocks_head.insert(0, head)
        # blocks_tail.append(nn.Dropout(0.7, inplace=False))
        blocks_tail.append(tail)
        blocks_tail.append(nn.Conv2d(2560, 512, 1, bias=False))  # projection
        self.backbone_head = nn.Sequential(*blocks_head)
        self.backbone_tail = nn.Sequential(*blocks_tail)

    def forward(self, x):
        aux = self.backbone_head(x)
        output = self.backbone_tail(aux)
        return aux, output


class ASPPConv(nn.Module):
    def __init__(self, inplanes, outplanes, kernel_size, padding, dilation):
        super(ASPPConv, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, outplanes, kernel_size=kernel_size,
                                     stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)
        output = self.relu(x)
        return output


class ASPPPooling(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(ASPPPooling, self).__init__()
        self.globalavgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv2d(inplanes, outplanes, 1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.globalavgpool(x)
        x = self.conv(x)
        x = self.bn(x)
        output = self.relu(x)
        return output


class ASPP(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(ASPP, self).__init__()
        dilations = [1, 6, 12, 18]
        self.aspp1 = ASPPConv(inplanes, outplanes, 1,
                              padding=0, dilation=dilations[0])
        self.aspp2 = ASPPConv(inplanes, outplanes, 3,
                              padding=dilations[1], dilation=dilations[1])
        self.aspp3 = ASPPConv(inplanes, outplanes, 3,
                              padding=dilations[2], dilation=dilations[2])
        self.aspp4 = ASPPConv(inplanes, outplanes, 3,
                              padding=dilations[3], dilation=dilations[3])
        self.global_avg_pool = ASPPPooling(inplanes, outplanes)
        self.project = nn.Sequential(
            nn.Conv2d(outplanes*5, outplanes, 1, bias=False),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x.size()[
                           2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        output = self.project(x)
        return output


class DeepLabHead(nn.Sequential):
    def __init__(self, in_ch, out_ch, n_classes):
        super(DeepLabHead, self).__init__()
        self.add_module("0", ASPP(in_ch, out_ch))
        self.add_module("1", nn.Conv2d(
            out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False))
        self.add_module("2", nn.BatchNorm2d(out_ch))
        self.add_module("3", nn.ReLU())
        self.add_module("4", nn.Conv2d(
            out_ch, n_classes, kernel_size=1, stride=1))


class DeepLabV3(nn.Sequential):
    def __init__(self, n_classes, n_blocks, atrous_rates):
        super(DeepLabV3, self).__init__()
        self.backbone = EffNet()
        self.classifier = DeepLabHead(in_ch=512, out_ch=256, n_classes=12)

    def forward(self, x):
        h = self.backbone(x)
        h = self.classifier(h)
        output = F.interpolate(
            h, size=x.shape[2:], mode="bilinear", align_corners=False)
        return output


class DeepLabV3_plus(nn.Sequential):
    def __init__(self, n_classes, n_blocks, atrous_rates):
        super(DeepLabV3_plus, self).__init__()
        self.backbone = EffNet2()
        self.classifier = DeepLabHead(in_ch=512, out_ch=256, n_classes=12)
        self.aux_projection = nn.Conv2d(48, 12, 1, bias=False)
        self.conv = nn.Conv2d(24, 12, 3, stride=1, padding=1, bias=False)

    def forward(self, x):
        aux, h = self.backbone(x)
        h = self.classifier(h)
        aux = self.aux_projection(aux)
        temp_output = F.interpolate(
            h, size=(128, 128), mode="bilinear", align_corners=False)
        temp_output = torch.cat((temp_output, aux), dim=1)
        temp_output = self.conv(temp_output)
        output = F.interpolate(
            temp_output, size=x.shape[2:], mode="bilinear", align_corners=False)
        return output
