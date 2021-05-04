from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import url_map, get_model_params
import torch
import torch.nn as nn
import torch.nn.functional as F

class DecoderBlock(nn.Module):
    def __init__(self, inputs, outputs, concats, method):
        super(DecoderBlock, self).__init__()
        self.method = method
        # Using deconv method
        self.up_transpose = nn.ConvTranspose2d(
            inputs, outputs, kernel_size=2, stride=2)
        self.up_conv = nn.Sequential(
            nn.Conv2d(outputs + concats, outputs, kernel_size=3, padding=1),
            nn.BatchNorm2d(outputs),
            nn.ReLU(),
            nn.Conv2d(outputs, outputs, kernel_size=3, padding=1),
            nn.BatchNorm2d(outputs),
            nn.ReLU(),
        )

    def forward(self, x, x_copy):
        # print('Input:', x.shape, x_copy.shape)
        x = self.up_transpose(x)
        # print('Up:', x.shape)
        if self.method == 'interpolate':
            x = F.interpolate(x, size=(x_copy.size(2), x_copy.size(3)),
                              mode='bilinear', align_corners=True)
        else:
            # for different sizes
            diffX = x_copy.size()[3] - x.size()[3]
            diffY = x_copy.size()[2] - x.size()[2]
            x = F.pad(x, (diffX // 2, diffX - diffX // 2,
                          diffY // 2, diffX - diffY // 2))
        #print('Scale:', x.shape)

        # Concatenate
        x = torch.cat([x_copy, x], dim=1)
        #print('Concat', x.shape)
        x = self.up_conv(x)
        #print('UpConv:', x.shape)
        return x


class MiddleBlock(nn.Module):
    def __init__(self, inputs, outputs):
        super(MiddleBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inputs, outputs, kernel_size=3, padding=1),
            nn.BatchNorm2d(outputs),
            nn.ReLU(),
            nn.Conv2d(outputs, outputs, kernel_size=3, padding=1),
            nn.BatchNorm2d(outputs),
            nn.ReLU(),
        )

    def forward(self, x):
        #print('Input:', x.shape)
        x = self.conv(x)
        #print('Middle:', x.shape)
        return x


efficient_net_encoders = {
    "efficientnet-b0": {
        "out_channels": (3, 32, 24, 40, 112, 320),
        "stage_idxs": (3, 5, 9, 16),
        "model_name": "efficientnet-b0",
    },
    "efficientnet-b1": {
        "out_channels": (3, 32, 24, 40, 112, 320),
        "stage_idxs": (5, 8, 16, 23),
        "model_name": "efficientnet-b1",
    },
    "efficientnet-b2": {
        "out_channels": (3, 32, 24, 48, 120, 352),
        "stage_idxs": (5, 8, 16, 23),
        "model_name": "efficientnet-b2",
    },
    "efficientnet-b3": {
        "out_channels": (3, 40, 32, 48, 136, 384),
        "stage_idxs": (5, 8, 18, 26),
        "model_name": "efficientnet-b3",
    },
    "efficientnet-b4": {
        "out_channels": (3, 48, 32, 56, 160, 448),
        "stage_idxs": (6, 10, 22, 32),
        "model_name": "efficientnet-b4",
    },
    "efficientnet-b5": {
        "out_channels": (3, 48, 40, 64, 176, 512),
        "stage_idxs": (8, 13, 27, 39),
        "model_name": "efficientnet-b5",
    },
    "efficientnet-b6": {
        "out_channels": (3, 56, 40, 72, 200, 576),
        "stage_idxs": (9, 15, 31, 45),
        "model_name": "efficientnet-b6",
    },
    "efficientnet-b7": {
        "out_channels": (3, 64, 48, 80, 224, 640),
        "stage_idxs": (11, 18, 38, 55),
        "model_name": "efficientnet-b7",
    }
}


class EfficientNetEncoder(EfficientNet):
    def __init__(self, stage_idxs, out_channels, model_name, depth=5):

        blocks_args, global_params = get_model_params(
            model_name, override_params=None)
        super().__init__(blocks_args, global_params)

        self._stage_idxs = stage_idxs
        self._out_channels = out_channels
        self._depth = depth
        self._in_channels = 3

        del self._fc

    def get_stages(self):
        return [
            nn.Identity(),
            nn.Sequential(self._conv_stem, self._bn0, self._swish),
            self._blocks[:self._stage_idxs[0]],
            self._blocks[self._stage_idxs[0]:self._stage_idxs[1]],
            self._blocks[self._stage_idxs[1]:self._stage_idxs[2]],
            self._blocks[self._stage_idxs[2]:],
        ]

    def forward(self, x):
        stages = self.get_stages()

        block_number = 0.
        drop_connect_rate = self._global_params.drop_connect_rate

        features = []
        for i in range(self._depth + 1):
            # Identity and Sequential stages
            if i < 2:
                x = stages[i](x)
            # Block stages need drop_connect rate
            else:
                for module in stages[i]:
                    drop_connect = drop_connect_rate * \
                        block_number / len(self._blocks)
                    block_number += 1.
                    x = module(x, drop_connect)
            features.append(x)
        return features

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop("_fc.bias")
        state_dict.pop("_fc.weight")
        super().load_state_dict(state_dict, **kwargs)


class EfficientUNet(nn.Module):
    def __init__(self, nclasses, in_channels, version, method=''):
        super().__init__()
        self.init_conv = nn.Conv2d(in_channels, 3, kernel_size=1, padding=0)
        self.effnet = EfficientNetEncoder(
            **efficient_net_encoders[f'efficientnet-{version}'])
        self.middle_block = MiddleBlock(320, 640)
        self.up_conv = nn.ModuleList([
            DecoderBlock(640, 320, 320, method),
            DecoderBlock(320, 112, 112, method),
            DecoderBlock(112, 40, 40, method),
            DecoderBlock(40, 24, 24, method),
            DecoderBlock(24, 32, 32, method),
            DecoderBlock(32, 64, 3, method)
        ])
        self.final_conv = nn.Conv2d(64, nclasses, kernel_size=1)

    def forward(self, x):
        x = self.init_conv(x)
        enc_fts = self.effnet(x)
        x = self.middle_block(enc_fts[-1])
        enc_fts = enc_fts[::-1]
        for enc_ft, upconv in zip(enc_fts, self.up_conv):
            x = upconv(x, enc_ft)
        return self.final_conv(x)



def conv3x3_relu(in_ch, out_ch, rate=1):
    conv3x3_relu = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=3,
                                           stride=1, padding=rate, dilation=rate),
                                 nn.ReLU())
    return conv3x3_relu


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.features = nn.Sequential(conv3x3_relu(3, 64),
                                      conv3x3_relu(64, 64),
                                      nn.MaxPool2d(3, stride=2, padding=1),
                                      conv3x3_relu(64, 128),
                                      conv3x3_relu(128, 128),
                                      nn.MaxPool2d(3, stride=2, padding=1),
                                      conv3x3_relu(128, 256),
                                      conv3x3_relu(256, 256),
                                      conv3x3_relu(256, 256),
                                      nn.MaxPool2d(3, stride=2, padding=1),
                                      conv3x3_relu(256, 512),
                                      conv3x3_relu(512, 512),
                                      conv3x3_relu(512, 512),
                                      nn.MaxPool2d(3, stride=1, padding=1),  # 마지막 stride=1로 해서 두 layer 크기 유지
                                      # and replace subsequent conv layer r=2
                                      conv3x3_relu(512, 512, rate=2),
                                      conv3x3_relu(512, 512, rate=2),
                                      conv3x3_relu(512, 512, rate=2),
                                      nn.MaxPool2d(3, stride=1, padding=1))  # 마지막 stride=1로 해서 두 layer 크기 유지

    def forward(self, x):
        out = self.features(x)
        return out



class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=1024, num_classes=21):
        super(ASPP, self).__init__()
        # atrous 3x3, rate=6
        self.conv_3x3_r6 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=6, dilation=6)
        # atrous 3x3, rate=12
        self.conv_3x3_r12 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=12, dilation=12)
        # atrous 3x3, rate=18
        self.conv_3x3_r18 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=18, dilation=18)
        # atrous 3x3, rate=24
        self.conv_3x3_r24 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=24, dilation=24)
        self.drop_conv_3x3 = nn.Dropout2d(0.5)

        self.conv_1x1 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.drop_conv_1x1 = nn.Dropout2d(0.5)

        self.conv_1x1_out = nn.Conv2d(out_channels, num_classes, kernel_size=1)

    def forward(self, feature_map):
        # 1번 branch
        # shape: (batch_size, out_channels, height/output_stride, width/output_stride)
        out_3x3_r6 = self.drop_conv_3x3(F.relu(self.conv_3x3_r6(feature_map)))
        out_img_r6 = self.drop_conv_1x1(F.relu(self.conv_1x1(out_3x3_r6)))
        out_img_r6 = self.conv_1x1_out(out_img_r6)
        # 2번 branch
        # shape: (batch_size, out_channels, height/output_stride, width/output_stride)
        out_3x3_r12 = self.drop_conv_3x3(F.relu(self.conv_3x3_r12(feature_map)))
        out_img_r12 = self.drop_conv_1x1(F.relu(self.conv_1x1(out_3x3_r12)))
        out_img_r12 = self.conv_1x1_out(out_img_r12)
        # 3번 branch
        # shape: (batch_size, out_channels, height/output_stride, width/output_stride)
        out_3x3_r18 = self.drop_conv_3x3(F.relu(self.conv_3x3_r18(feature_map)))
        out_img_r18 = self.drop_conv_1x1(F.relu(self.conv_1x1(out_3x3_r18)))
        out_img_r18 = self.conv_1x1_out(out_img_r18)
        # 4번 branch
        # shape: (batch_size, out_channels, height/output_stride, width/output_stride)
        out_3x3_r24 = self.drop_conv_3x3(F.relu(self.conv_3x3_r24(feature_map)))
        out_img_r24 = self.drop_conv_1x1(F.relu(self.conv_1x1(out_3x3_r24)))
        out_img_r24 = self.conv_1x1_out(out_img_r24)

        out = sum([out_img_r6, out_img_r12, out_img_r18, out_img_r24])

        return out


class DeepLabV2(nn.Module):
    ## VGG 위에 ASPP 쌓기
    def __init__(self, backbone, classifier, upsampling=8):
        super(DeepLabV2, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.upsampling = upsampling

    def forward(self, x):
        x = self.backbone(x)
        print(x.size())
        _, _, feature_map_h, feature_map_w = x.size()
        x = self.classifier(x)
        out = F.interpolate(x, size=(feature_map_h * self.upsampling, feature_map_w * self.upsampling), mode="bilinear")
        return out

class EffNet(nn.Module):
    def __init__(self):
        super(EffNet, self).__init__()
        effnet = EfficientNet.from_pretrained('efficientnet-b5')
        head = nn.Sequential(effnet._conv_stem, effnet._bn0)
        blocks = list(effnet._blocks.children())
        tail = nn.Sequential(effnet._conv_head, effnet._bn1)
        blocks.insert(0, head)
        blocks.append(tail)
        blocks.append(nn.Conv2d(2048, 512, 1, bias=False)) # projection
        self.backbone = nn.Sequential(*blocks)
    def forward(self, x):
        output = self.backbone(x)
        print(output.shape)
        return output