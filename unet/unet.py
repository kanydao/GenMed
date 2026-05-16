import torch
import torch.nn as nn

def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()

def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU', stride=1):
    layers = []
    layers.append(ConvBatchNorm(in_channels, out_channels, activation, stride))

    for _ in range(nb_Conv - 1):
        layers.append(ConvBatchNorm(out_channels, out_channels, activation))
    return nn.Sequential(*layers)

class ConvBatchNorm(nn.Module):
    def __init__(self, in_channels, out_channels, activation='ReLU', stride=1):
        super(ConvBatchNorm, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.norm = nn.BatchNorm3d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)

class ConvGroupNorm(nn.Module):
    def __init__(self, in_channels, out_channels, activation='ReLU', stride=1, num_groups=8):
        super(ConvGroupNorm, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(DownBlock, self).__init__()
        self.maxpool = nn.MaxPool3d(2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x):
        out = self.maxpool(x)
        return self.nConvs(out)

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(UpBlock, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x, skip_x):
        x = self.up(x)
        x = torch.cat([x, skip_x], dim=1)
        return self.nConvs(x)

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, channels, nb_Conv=2, activation='ReLU'):
        super(UNet, self).__init__()
        self.nb_Conv = nb_Conv
        self.activation = activation

        self.inc = ConvBatchNorm(in_channels, channels[0], activation=activation)
        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        for i in range(len(channels) - 1):
            self.down_blocks.append(DownBlock(channels[i], channels[i + 1], nb_Conv, activation=activation))

        for i in range(len(channels) - 1, 0, -1):
            self.up_blocks.append(UpBlock(channels[i] + channels[i - 1], channels[i - 1], nb_Conv, activation=activation))

        self.outc = nn.Conv3d(channels[0], out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        downs = [x1]

        for down_block in self.down_blocks:
            x1 = down_block(x1)
            downs.append(x1)

        for i, up_block in enumerate(self.up_blocks):
            x1 = up_block(x1, downs[-(i + 2)])

        logits = self.outc(x1)
        return logits

if __name__ == '__main__':
    in_channels = 1
    out_channels = 1
    channels = [32, 64, 128, 256, 512]
    model = UNet(in_channels, out_channels, channels)
    x = torch.randn(1, in_channels, 64, 64, 64)
    output = model(x)
    print(output.shape)
