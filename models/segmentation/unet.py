import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.seq(x)

class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.d1 = DoubleConv(3, 32)
        self.d2 = DoubleConv(32, 64)
        self.d3 = DoubleConv(64, 128)
        self.d4 = DoubleConv(128, 256)

        self.pool = nn.MaxPool2d(2)
        self.mid = DoubleConv(256, 512)

        self.u4 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.c4 = DoubleConv(512, 256)

        self.u3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.c3 = DoubleConv(256, 128)

        self.u2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.c2 = DoubleConv(128, 64)

        self.u1 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.c1 = DoubleConv(64, 32)

        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(self.pool(d1))
        d3 = self.d3(self.pool(d2))
        d4 = self.d4(self.pool(d3))

        m = self.mid(self.pool(d4))

        x = self.c4(torch.cat([self.u4(m), d4], dim=1))
        x = self.c3(torch.cat([self.u3(x), d3], dim=1))
        x = self.c2(torch.cat([self.u2(x), d2], dim=1))
        x = self.c1(torch.cat([self.u1(x), d1], dim=1))

        return torch.sigmoid(self.out(x))