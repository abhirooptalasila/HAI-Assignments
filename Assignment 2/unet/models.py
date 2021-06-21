import torch
import logger
log = logger.get_logger(__name__)

import torch.nn as nn

def double_conv(in_c, out_c):
    conv = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3), 
        nn.ReLU(inplace=True), 
        nn.Conv2d(out_c, out_c, kernel_size=3), 
        nn.ReLU(inplace=True)
    )
    return conv

def up_trans_cov(in_c, out_c, x, cat):
    up_trans = nn.ConvTranspose2d(
            in_channels=in_c, out_channels=out_c, 
            kernel_size=2, stride=2
    )
    up_conv = double_conv(in_c, out_c)

    x1 = up_trans(x)
    cropped = crop_tensor(cat, x1)
    x2 = up_conv(torch.cat([cropped, x1], 1))
    return x2

def crop_tensor(tensor, target_tensor):
    tensor_size = tensor.size()[2]
    target_size = target_tensor.size()[2] 
    delta = tensor_size - target_size
    delta = delta // 2
    return tensor[:, :, delta:tensor_size-delta, delta:tensor_size-delta]

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_conv_1 = double_conv(1, 64)
        self.down_conv_2 = double_conv(64, 128)
        self.down_conv_3 = double_conv(128, 256)
        self.down_conv_4 = double_conv(256, 512)
        self.down_conv_5 = double_conv(512, 1024)

        self.out = nn.Conv2d(
            in_channels=64, out_channels=2, 
            kernel_size=1
        )

    def forward(self, x):
        # (b, c, h, w)
        # encoder part
        log.debug("Encoder I/P Size: {}".format(x.size()))
        x1 = self.down_conv_1(x) #1
        x2 = self.max_pool_2x2(x1)
        x3 = self.down_conv_2(x2) #2
        x4 = self.max_pool_2x2(x3)
        x5 = self.down_conv_3(x4) #3
        x6 = self.max_pool_2x2(x5)
        x7 = self.down_conv_4(x6) #4
        x8 = self.max_pool_2x2(x7)
        x9 = self.down_conv_5(x8)
        # log.debug("Encoder O/P Size: {}".format(x9.size()))

        # decoder part
        x10 = up_trans_cov(1024, 512, x9, x7)
        x11 = up_trans_cov(512, 256, x10, x5)
        x12 = up_trans_cov(256, 128, x11, x3)
        x13 = up_trans_cov(128, 64, x12, x1)
        output = self.out(x13)

        log.debug("Decoder O/P Size: {}".format(output.size()))
        return output

    def predict(self):
        pass