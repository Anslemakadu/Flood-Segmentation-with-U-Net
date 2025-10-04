import torch
import torch.nn as nn


# A compact U-Net implementation suitable for CPU and small experiments.
# Keep it simple so you can explain every block.


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down, self).__init__()
        self.pool = nn.MaxPool2d(2)
        self.dc = DoubleConv(in_ch, out_ch)
    def forward(self, x):
            x = self.pool(x)
            x = self.dc(x)
            return x


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Up, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x, skip):
        x = self.up(x)
        # pad if sizes mismatch
        diffY = skip.size()[2] - x.size()[2]
        diffX = skip.size()[3] - x.size()[3]
        if diffY !=0 or diffX !=0:
            x = nn.functional.pad(x, [diffX//2, diffX - diffX//2, diffY//2, diffY - diffY//2])
        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        return x


class UNetSmall(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_c=32):
        super(UNetSmall, self).__init__()
        self.in_conv = DoubleConv(in_channels, base_c)
        self.down1 = Down(base_c, base_c*2)
        self.down2 = Down(base_c*2, base_c*4)
        self.down3 = Down(base_c*4, base_c*8)
        self.bottleneck = DoubleConv(base_c*8, base_c*16)
        self.up3 = Up(base_c*16 + base_c*8, base_c*8)
        self.up2 = Up(base_c*8 + base_c*4, base_c*4)
        self.up1 = Up(base_c*4 + base_c*2, base_c*2)
        self.final_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.final_conv = nn.Conv2d(base_c*2 + base_c, out_channels, kernel_size=1)
        
    def forward(self, x):
        x1 = self.in_conv(x) # [B, base_c, H, W]
        x2 = self.down1(x1) # [B, base_c*2, H/2, W/2]
        x3 = self.down2(x2) # [B, base_c*4, H/4, W/4]
        x4 = self.down3(x3) # [B, base_c*8, H/8, W/8]
        b = self.bottleneck(x4) # [B, base_c*16, H/8, W/8]
        u3 = self.up3(b, x4)
        u2 = self.up2(u3, x3)
        u1 = self.up1(u2, x2)
        u0 = self.final_up(u1)
        out = torch.cat([u0, x1], dim=1)
        out = self.final_conv(out)
        return out