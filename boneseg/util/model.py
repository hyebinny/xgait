import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

class AttentionGate(nn.Module):
    def __init__(self, g_ch, x_ch, inter_ch):
        super().__init__()
        self.Wg = nn.Sequential(
            nn.Conv2d(g_ch, inter_ch, 1, bias=False),
            nn.BatchNorm2d(inter_ch),
        )
        self.Wx = nn.Sequential(
            nn.Conv2d(x_ch, inter_ch, 1, bias=False),
            nn.BatchNorm2d(inter_ch),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_ch, 1, 1, bias=True),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # g: gating (decoder), x: skip (encoder)
        attn = self.relu(self.Wg(g) + self.Wx(x))
        attn = self.psi(attn)
        return x * attn

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)

    def forward(self, x):
        return self.up(x)

class AttentionUNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=7, base=64):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, base)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(base, base*2)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(base*2, base*4)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = ConvBlock(base*4, base*8)
        self.pool4 = nn.MaxPool2d(2)

        self.bott = ConvBlock(base*8, base*16)

        self.up4 = Up(base*16, base*8)
        self.att4 = AttentionGate(g_ch=base*8, x_ch=base*8, inter_ch=base*4)
        self.dec4 = ConvBlock(base*16, base*8)

        self.up3 = Up(base*8, base*4)
        self.att3 = AttentionGate(g_ch=base*4, x_ch=base*4, inter_ch=base*2)
        self.dec3 = ConvBlock(base*8, base*4)

        self.up2 = Up(base*4, base*2)
        self.att2 = AttentionGate(g_ch=base*2, x_ch=base*2, inter_ch=base)
        self.dec2 = ConvBlock(base*4, base*2)

        self.up1 = Up(base*2, base)
        self.att1 = AttentionGate(g_ch=base, x_ch=base, inter_ch=base//2)
        self.dec1 = ConvBlock(base*2, base)

        self.head = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)      # [B,base,H,W]
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b  = self.bott(self.pool4(e4))

        d4 = self.up4(b)
        s4 = self.att4(d4, e4)
        d4 = self.dec4(torch.cat([d4, s4], dim=1))

        d3 = self.up3(d4)
        s3 = self.att3(d3, e3)
        d3 = self.dec3(torch.cat([d3, s3], dim=1))

        d2 = self.up2(d3)
        s2 = self.att2(d2, e2)
        d2 = self.dec2(torch.cat([d2, s2], dim=1))

        d1 = self.up1(d2)
        s1 = self.att1(d1, e1)
        d1 = self.dec1(torch.cat([d1, s1], dim=1))

        return self.head(d1)
