import torch
import torch.nn as nn
import torch.nn.functional as F

class FACNN(nn.Module):
    """Some Information about FACNN"""
    def __init__(self,scale):
        super(FACNN, self).__init__()
        self.scale = scale
        self.conv1 = nn.Sequential(
                nn.Conv2d(3,64,kernel_size=5, stride=1, padding=2, bias=True),
                nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64,32,kernel_size=1, stride=1, padding=0),
            nn.Conv2d(32,12,kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(12,12,kernel_size=3, stride=1, padding=1),
            nn.Conv2d(12,3,kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
    def forward(self, x):
        # x = F.interpolate(x,scale_factor=(self.scale, self.scale), mode='bicubic',align_corners=True)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = F.interpolate(out,scale_factor=(self.scale, self.scale), mode='bicubic',align_corners=True)
        # out = out.squeeze(0).permute(1,2,0).mul(255.0).clamp(0,255.0).type(torch.ByteTensor)
        return out