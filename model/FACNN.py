import torch
import torch.nn as nn
import torch.nn.functional as F
class LDSR_V4(nn.Module):
    def __init__(self, scale_factor, main_blocks=10, sub_blocks=3, num_channels=3, num_feats=128):
        super(LDSR_V4, self).__init__()
        act = nn.ReLU(True)
        n_feats = num_feats
        n_shrinks = num_feats//2
        self.scale = scale_factor
        self.first_part = nn.Sequential(nn.Conv2d(num_channels, n_feats, kernel_size=5, padding=5//2),nn.PReLU(n_feats))
        self.first_res = []
        for i in range(sub_blocks):
            self.first_res.extend([ResBlock(
                    n_feats, res_scale=1.0
                )])
        self.first_res = nn.Sequential(*self.first_res)
        self.mid_part =  nn.Sequential(nn.Conv2d(n_feats, n_shrinks, kernel_size=1), nn.PReLU(n_shrinks))
        self.mid_res = []
        for i in range(main_blocks):
            self.mid_res.extend([ResBlock(
                    n_shrinks, res_scale=1.0
                )])
        self.mid_res = nn.Sequential(*self.mid_res)
        
        self.last_part =  nn.Sequential(nn.Conv2d(n_shrinks, n_feats, kernel_size=1), nn.PReLU(n_feats))
        self.last_res = []
        for i in range(sub_blocks):
            self.last_res.extend([ResBlock(
                    n_feats, res_scale=1.0
                )])
        self.last_res = nn.Sequential(*self.last_res)
        self.upsample = nn.Sequential(
            nn.Conv2d(n_feats, num_channels * (scale_factor ** 2), kernel_size=3, stride=1, padding=1), 
            nn.PixelShuffle(scale_factor),
            )
    def forward(self, x):
        x = self.first_part(x)
        res = self.first_res(x)
        res += x
        x = self.mid_part(res)
        res = self.mid_res(x)
        res += x
        x = self.last_part(res)
        res = self.last_res(x)
        res += x
        x = self.upsample(res)
        return x
class LDSR_V4_1(nn.Module):
    def __init__(self, scale_factor, main_blocks=10, sub_blocks=3, num_channels=3, num_feats=128):
        super(LDSR_V4_1, self).__init__()
        n_feats = num_feats
        n_shrinks = num_feats//2
        self.scale = scale_factor
        self.sf1 = nn.Sequential(nn.Conv2d(num_channels, n_feats, kernel_size=3, padding=3//2),nn.PReLU(n_feats))
        self.sf2 = nn.Sequential(nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=3//2),nn.PReLU(n_feats))
        self.first_res = []
        for i in range(sub_blocks):
            self.first_res .extend([ResBlock(
                    n_feats, res_scale=1.0
                )])
        self.first_res = nn.Sequential(*self.first_res)
        self.mid_part =  nn.Sequential(nn.Conv2d(n_feats, n_shrinks, kernel_size=1), nn.PReLU(n_shrinks))
        self.mid_res = []
        for i in range(main_blocks):
            self.mid_res .extend([ResBlock(
                    n_shrinks, res_scale=1.0
                )])
        self.mid_res = nn.Sequential(*self.mid_res)
        
        self.last_part =  nn.Sequential(nn.Conv2d(n_shrinks, n_feats, kernel_size=1), nn.PReLU(n_feats))
        self.last_res = []
        for i in range(sub_blocks):
            self.last_res .extend([ResBlock(
                    n_feats, res_scale=1.0
                )])
        self.last_res = nn.Sequential(*self.last_res)
        self.upsample = nn.Sequential(
            nn.Conv2d(n_feats, num_channels * (scale_factor ** 2), kernel_size=3, stride=1, padding=1), 
            nn.PixelShuffle(scale_factor),
            nn.ReLU(True)
            )
    def forward(self, x):
        x = self.sf1(x)
        x = self.sf2(x)
        res = self.first_res(x)
        res += x
        x = self.mid_part(res)
        res = self.mid_res(x)
        res += x
        x = self.last_part(res)
        res = self.last_res(x)
        res += x
        x = self.upsample(res)
        return x
class ResBlock(nn.Module):
    def __init__(self, n_feats, res_scale=1.0):
        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(nn.Conv2d(n_feats, n_feats, kernel_size=3, bias=True, padding=3//2))
            if i == 0:
                m.append(nn.ReLU(True))
        self.body = nn.Sequential(*m)
        self.res_scale = res_scale
    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res

class FACNN(nn.Module):
    """Some Information about FACNN"""
    def __init__(self,scale):
        super(FACNN, self).__init__()
        self.scale = scale
        self.conv1 = nn.Sequential(
                nn.Conv2d(3,64,kernel_size=5, stride=1, padding=2, bias=True),
                nn.Conv2d(64,128,kernel_size=5, stride=1, padding=2, bias=True),
                nn.Conv2d(128,18,kernel_size=5, stride=1, padding=2, bias=True),
                nn.BatchNorm2d(18),
                nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(18,64,kernel_size=1, stride=1, padding=0),
            nn.Conv2d(64,128,kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.ConvTranspose2d(128,64,kernel_size=9, stride=1, padding=9//2),
            nn.ConvTranspose2d(64,32,kernel_size=5, stride=1, padding=2),
            nn.ConvTranspose2d(32,18,kernel_size=3, stride=1, padding=1),
            nn.ReLU()   
        )
        self.deconv = nn.ConvTranspose2d(18,3,kernel_size=3, stride=2, padding=1,output_padding=1)
    def forward(self, x):
        out = self.conv1(x)
        conv1 = out
        out = self.conv2(out)
        out = self.conv3(out)
        out = out+conv1
        out = self.deconv(out)
        # out = out.squeeze(0).permute(1,2,0).mul(255.0).clamp(0,255.0).type(torch.ByteTensor)
        return out