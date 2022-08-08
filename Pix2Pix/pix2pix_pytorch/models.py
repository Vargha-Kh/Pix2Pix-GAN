import torch.nn as nn
import torch.nn.functional as F
import torch


def weights_init_normal(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)


class DownSampleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, strides=2, padding=1, activation=True, batchnorm=True,
                 dropout_rate=0.0, inplace=False):
        super(DownSampleConv, self).__init__()
        self.activation = activation
        self.batchnorm = batchnorm
        self.dropout_rate = dropout_rate
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=strides, padding=padding)

        if batchnorm:
            self.batch_normalization = nn.InstanceNorm2d(out_channels)
        if activation:
            self.activation_f = nn.LeakyReLU(0.2, inplace=inplace)
        if dropout_rate:
            self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.conv(x)
        if self.batchnorm:
            x = self.batch_normalization(x)
        if self.activation:
            x = self.activation_f(x)
        if self.dropout_rate:
            x = self.dropout(x)

        return x


class UpSampleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, strides=2, padding=1, dropout_rate=0.0):
        super(UpSampleConv, self).__init__()
        self.dropout_rate = dropout_rate
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=strides, padding=padding,
                               bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(True),
        )
        if dropout_rate:
            self.drop = nn.Dropout2d(dropout_rate)

    def forward(self, x, skip_input):
        x = self.deconv(x)
        if self.dropout_rate:
            x = self.drop(x)
        x = torch.cat((x, skip_input), 1)
        return x


class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(Generator, self).__init__()
        self.down1 = DownSampleConv(in_channels, 64, batchnorm=False)
        self.down2 = DownSampleConv(64, 128)
        self.down3 = DownSampleConv(128, 256)
        self.down4 = DownSampleConv(256, 512, dropout_rate=0.5)
        self.down5 = DownSampleConv(512, 512, dropout_rate=0.5)
        self.down6 = DownSampleConv(512, 512, dropout_rate=0.5)
        self.down7 = DownSampleConv(512, 512, dropout_rate=0.5)
        self.down8 = DownSampleConv(512, 512, batchnorm=False, dropout_rate=0.5)

        self.up1 = UpSampleConv(512, 512, dropout_rate=0.5)
        self.up2 = UpSampleConv(1024, 512, dropout_rate=0.5)
        self.up3 = UpSampleConv(1024, 512, dropout_rate=0.5)
        self.up4 = UpSampleConv(1024, 512, dropout_rate=0.5)
        self.up5 = UpSampleConv(1024, 256)
        self.up6 = UpSampleConv(512, 128)
        self.up7 = UpSampleConv(256, 64)

        self.last_conv = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        ds1 = self.down1(x)
        ds2 = self.down2(ds1)
        ds3 = self.down3(ds2)
        ds4 = self.down4(ds3)
        ds5 = self.down5(ds4)
        ds6 = self.down6(ds5)
        ds7 = self.down7(ds6)
        ds8 = self.down8(ds7)
        us1 = self.up1(ds8, ds7)
        us2 = self.up2(us1, ds6)
        us3 = self.up3(us2, ds5)
        us4 = self.up4(us3, ds4)
        us5 = self.up5(us4, ds3)
        us6 = self.up6(us5, ds2)
        us7 = self.up7(us6, ds1)
        return self.last_conv(us7)


class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            DownSampleConv(in_channels + in_channels, 64, batchnorm=False, inplace=True),
            DownSampleConv(64, 128, inplace=True),
            DownSampleConv(128, 256, inplace=True),
            DownSampleConv(256, 512, inplace=True),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, (4, 4), padding=1, bias=False)
        )

    def forward(self, x, y):
        img_input = torch.cat([x, y], 1)
        return self.model(img_input)
