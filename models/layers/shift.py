import torch.nn as nn
from torchshifts import Shift2d


class ShiftConv2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ShiftConv2d, self).__init__()

        self.shift = Shift2d(in_channels)
        self.act = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x, _ = self.shift(x)
        return self.conv2(self.act(x))


if __name__ == '__main__':
    import torch

    shift_conv = ShiftConv2d(3, 10)

    data = torch.rand(1, 3, 32, 32)
    b = shift_conv(data)
    print(b.size())