import logger
log = logger.get_logger(__name__)

import torch
import numpy as np
import torch.nn as nn
from torchvision import models

class Double_conv(nn.Module):
    def __init__(self, in_c, out_c):
        super(Double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, stride=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, stride=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class Conv_down(nn.Module):
    def __init__(self, in_c, out_c):
        super(Conv_down, self).__init__()
        self.conv = Double_conv(in_c, out_c)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        pool_x = self.pool(x)
        return pool_x, x

class Conv_up(nn.Module):
    def __init__(self, in_c, out_c):
        super(Conv_up, self).__init__()
        self.up = nn.ConvTranspose2d(
            in_c, out_c, kernel_size=2, stride=2)
        self.conv = Double_conv(in_c, out_c)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x2 = extract_img(x1, x2)
        x1 = torch.cat((x1, x2), dim=1)
        x1 = self.conv(x1)
        return x1


def extract_img(target, in_tensor):
    size = target.size()[2]
    dim1, dim2 = in_tensor.size()[2:]
    in_tensor = in_tensor[:, :, int((dim1-size)/2):int((dim1+size)/2),
                          int((dim2-size)/2):int((dim2+size)/2)]
    return in_tensor


class UNet(nn.Module):
    def __init__(self, in_c, out_c, scale_factor=1):
        super(UNet, self).__init__()
        filters = [64, 128, 256, 512, 1024]
        if type(scale_factor) == list:
            filters = scale_factor
        elif scale_factor != 1:
            filters = np.divide(filters, scale_factor).astype(int)

        self.encoder_1 = Conv_down(in_c, filters[0])
        self.encoder_2 = Conv_down(filters[0], filters[1])
        self.encoder_3 = Conv_down(filters[1], filters[2])
        self.encoder_4 = Conv_down(filters[2], filters[3])
        self.encoder_5 = Conv_down(filters[3], filters[4])
        self.decoder_1 = Conv_up(filters[4], filters[3])
        self.decoder_2 = Conv_up(filters[3], filters[2])
        self.decoder_3 = Conv_up(filters[2], filters[1])
        self.decoder_4 = Conv_up(filters[1], filters[0])
        self.out = nn.Conv2d(filters[0], out_c, 1)

    def forward(self, x):
        #log.debug("I/P Size: {}".format(x.size()))
        x, conv1 = self.encoder_1(x)
        x, conv2 = self.encoder_2(x)
        x, conv3 = self.encoder_3(x)
        x, conv4 = self.encoder_4(x)
        _, x = self.encoder_5(x)

        x = self.decoder_1(x, conv4)
        x = self.decoder_2(x, conv3)
        x = self.decoder_3(x, conv2)
        x = self.decoder_4(x, conv1)
        x = self.out(x)
        # x = torch.sigmoid(x)
        #log.debug("O/P Size: {}".format(x.size()))
        return x


def convrelu(in_c, out_c, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )


class ResNetUNet(nn.Module):
    def __init__(self, in_c, out_c,):
        super(ResNetUNet, self).__init__()

        self.base_model = models.resnet18(pretrained=True)
        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(512, 512, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)

        self.conv_original_size0 = convrelu(in_c, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)

        self.conv_last = nn.Conv2d(64, out_c, 1)

    def forward(self, x):
        x_original = self.conv_original_size0(x)
        x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        layer4 = self.layer4_1x1(layer4)
        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        out = self.conv_last(x)

        return out

class Conv_up_nested(nn.Module):
    def __init__(self, up_c, in_c, out_c):
        super(Conv_up_nested, self).__init__()
        self.up = nn.ConvTranspose2d(
            up_c, up_c, kernel_size=2, stride=2)
        self.conv = Double_conv(in_c, out_c)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat([x1, *x2], dim=1)
        x1 = self.conv(x1)
        return x1


class UNetplus(nn.Module):
    def __init__(self, in_c, out_c, scale_factor=1, ds=False):
        super(UNetplus, self).__init__()
        self.deep_supervision = ds
        filters = [64, 128, 256, 512, 1024]
        if type(scale_factor) == list:
            filters = scale_factor
        elif scale_factor != 1:
            filters = np.divide(filters, scale_factor).astype(int)

        self.encoder0_0 = Conv_down(in_c, filters[0])
        self.encoder1_0 = Conv_down(filters[0], filters[1])
        self.encoder2_0 = Conv_down(filters[1], filters[2])
        self.encoder3_0 = Conv_down(filters[2], filters[3])
        self.encoder4_0 = Conv_down(filters[3], filters[4])

        self.decoder0_1 = Conv_up_nested(
            filters[1], filters[0] + filters[1], filters[0])
        self.decoder1_1 = Conv_up_nested(
            filters[2], filters[1] + filters[2], filters[1])
        self.decoder2_1 = Conv_up_nested(
            filters[3], filters[2] + filters[3], filters[2])
        self.decoder3_1 = Conv_up_nested(
            filters[4], filters[3] + filters[4], filters[3])

        self.decoder0_2 = Conv_up_nested(
            filters[1], filters[0] + filters[1], filters[0])
        self.decoder1_2 = Conv_up_nested(
            filters[2], filters[1] + filters[2], filters[1])
        self.decoder2_2 = Conv_up_nested(
            filters[3], filters[2] + filters[3], filters[2])
        
        self.decoder0_3 = Conv_up_nested(
            filters[1], filters[0] + filters[1], filters[0])
        self.decoder1_3 = Conv_up_nested(
            filters[2], filters[1] + filters[2], filters[1])

        self.decoder0_4 = Conv_up_nested(
            filters[1], filters[0] + filters[1], filters[0])
        
        if self.deep_supervision:
            self.out1 = nn.Conv2d(filters[0], out_c, 1)
            self.out2 = nn.Conv2d(filters[0], out_c, 1)
            self.out3 = nn.Conv2d(filters[0], out_c, 1)
            self.out4 = nn.Conv2d(filters[0], out_c, 1)
        else:
            self.out = nn.Conv2d(filters[0], out_c, 1)

    def forward(self, x):
        #pool, (input + conv down)
        #log.debug("I/P Size: {}".format(x.size()))
        x0_0p, x0_0 = self.encoder0_0(x)
        x1_0p, x1_0 = self.encoder1_0(x0_0p)
        x0_1 = self.decoder0_1(x1_0, [x0_0])

        x2_0p, x2_0 = self.encoder2_0(x1_0p)
        x1_1 = self.decoder1_1(x2_0, [x1_0])
        x0_2 = self.decoder0_2(x1_1, [x0_1])

        x3_0p, x3_0 = self.encoder3_0(x2_0p)
        x2_1 = self.decoder2_1(x3_0, [x2_0])
        x1_2 = self.decoder1_2(x2_1, [x1_1])
        x0_3 = self.decoder0_3(x1_2, [x0_2])

        x4_0p, x4_0 = self.encoder4_0(x3_0p)
        x3_1 = self.decoder3_1(x4_0, [x3_0])
        x2_2 = self.decoder2_2(x3_1, [x2_1])
        x1_3 = self.decoder1_3(x2_2, [x1_2])
        x0_4 = self.decoder0_4(x1_3, [x0_3])

        if self.deep_supervision:
            out1 = self.out1(x0_1)
            out2 = self.out1(x0_2)
            out3 = self.out1(x0_3)
            out4 = self.out1(x0_4)
            return [out1, out2, out3, out4]
        else:
            out = self.out(x0_4)
        #log.debug("O/P Size: {}".format(out.size()))
        return out


class UNetplusplus(nn.Module):
    def __init__(self, in_c, out_c, scale_factor=1, ds=False):
        super(UNetplusplus, self).__init__()
        self.deep_supervision = ds
        filters = [64, 128, 256, 512, 1024]
        if type(scale_factor) == list:
            filters = scale_factor
        elif scale_factor != 1:
            filters = np.divide(filters, scale_factor).astype(int)

        self.encoder0_0 = Conv_down(in_c, filters[0])
        self.encoder1_0 = Conv_down(filters[0], filters[1])
        self.encoder2_0 = Conv_down(filters[1], filters[2])
        self.encoder3_0 = Conv_down(filters[2], filters[3])
        self.encoder4_0 = Conv_down(filters[3], filters[4])

        self.decoder0_1 = Conv_up_nested(
            filters[1], filters[0] + filters[1], filters[0])
        self.decoder1_1 = Conv_up_nested(
            filters[2], filters[1] + filters[2], filters[1])
        self.decoder2_1 = Conv_up_nested(
            filters[3], filters[2] + filters[3], filters[2])
        self.decoder3_1 = Conv_up_nested(
            filters[4], filters[3] + filters[4], filters[3])

        self.decoder0_2 = Conv_up_nested(
            filters[1], filters[0]*2 + filters[1], filters[0])
        self.decoder1_2 = Conv_up_nested(
            filters[2], filters[1]*2 + filters[2], filters[1])
        self.decoder2_2 = Conv_up_nested(
            filters[3], filters[2]*2 + filters[3], filters[2])
        
        self.decoder0_3 = Conv_up_nested(
            filters[1], filters[0]*3 + filters[1], filters[0])
        self.decoder1_3 = Conv_up_nested(
            filters[2], filters[1]*3 + filters[2], filters[1])

        self.decoder0_4 = Conv_up_nested(
            filters[1], filters[0]*4 + filters[1], filters[0])
        
        if self.deep_supervision:
            self.out1 = nn.Conv2d(filters[0], out_c, 1)
            self.out2 = nn.Conv2d(filters[0], out_c, 1)
            self.out3 = nn.Conv2d(filters[0], out_c, 1)
            self.out4 = nn.Conv2d(filters[0], out_c, 1)
        else:
            self.out = nn.Conv2d(filters[0], out_c, 1)

    def forward(self, x):
        #pool, (input + conv down)
        #log.debug("I/P Size: {}".format(x.size()))
        x0_0p, x0_0 = self.encoder0_0(x)
        x1_0p, x1_0 = self.encoder1_0(x0_0p)
        x0_1 = self.decoder0_1(x1_0, [x0_0])

        x2_0p, x2_0 = self.encoder2_0(x1_0p)
        x1_1 = self.decoder1_1(x2_0, [x1_0])
        x0_2 = self.decoder0_2(x1_1, [x0_0, x0_1])

        x3_0p, x3_0 = self.encoder3_0(x2_0p)
        x2_1 = self.decoder2_1(x3_0, [x2_0])
        x1_2 = self.decoder1_2(x2_1, [x1_0, x1_1])
        x0_3 = self.decoder0_3(x1_2, [x0_0, x0_1, x0_2])

        x4_0p, x4_0 = self.encoder4_0(x3_0p)
        x3_1 = self.decoder3_1(x4_0, [x3_0])
        x2_2 = self.decoder2_2(x3_1, [x2_0, x2_1])
        x1_3 = self.decoder1_3(x2_2, [x1_0, x1_1, x1_2])
        x0_4 = self.decoder0_4(x1_3, [x0_0, x0_1, x0_2, x0_3])

        if self.deep_supervision:
            out1 = self.out1(x0_1)
            out2 = self.out1(x0_2)
            out3 = self.out1(x0_3)
            out4 = self.out1(x0_4)
            return [out1, out2, out3, out4]
        else:
            out = self.out(x0_4)
        #log.debug("O/P Size: {}".format(out.size()))
        return out
        # x = torch.sigmoid(x)