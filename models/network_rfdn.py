import torch
import torch.nn as nn
import models.block_rfdn as B

def make_model(args, parent=False):
    model = RFDN()
    return model


class RFDN(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nc=50, nb=2, upscale=4):
        super(RFDN, self).__init__()

        self.fea_conv = B.conv_layer(in_nc, nc, kernel_size=3)

        self.B1 = B.RFDB(in_channels=nc)
        self.B2 = B.RFDB(in_channels=nc)
        self.c = B.conv_block(nc * nb, nc, kernel_size=1, act_type='lrelu')

        self.LR_conv = B.conv_layer(nc, nc, kernel_size=3)

        upsample_block = B.pixelshuffle_block
        self.upsampler = upsample_block(nc, out_nc, upscale_factor=upscale)
        self.scale_idx = 0


    def forward(self, input):
        out_fea = self.fea_conv(input)
        out_B1 = self.B1(out_fea)
        out_B2 = self.B2(out_B1)

        out_B = self.c(torch.cat([out_B1, out_B2], dim=1))
        out_lr = self.LR_conv(out_B) + out_fea

        output = self.upsampler(out_lr)

        return output

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx

