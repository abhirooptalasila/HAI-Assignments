import logger
log = logger.get_logger(__name__)

import torch
import torch.nn as nn

class Double_conv(nn.Module):
    def __init__(self, in_c, out_c):
        super(Double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, stride=1),
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
    def __init__(self, in_c, out_c):
        super(UNet, self).__init__()
        self.encoder_1 = Conv_down(in_c, 64)
        self.encoder_2 = Conv_down(64, 128)
        self.encoder_3 = Conv_down(128, 256)
        self.encoder_4 = Conv_down(256, 512)
        self.encoder_5 = Conv_down(512, 1024)
        self.decoder_1 = Conv_up(1024, 512)
        self.decoder_2 = Conv_up(512, 256)
        self.decoder_3 = Conv_up(256, 128)
        self.decoder_4 = Conv_up(128, 64)
        self.out = nn.Conv2d(64, out_c, 1)

    def forward(self, x):
        log.debug("I/P Size: {}".format(x.size()))
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
        # x = nn.Softmax(dim=1)(x)
        log.debug("O/P Size: {}".format(x.size()))
        return x