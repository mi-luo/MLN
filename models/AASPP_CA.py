import torch.nn as nn
import torch
import torch.nn.functional as F
from functools import reduce

# CA
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)
    def forward(self, x):
        return self.relu(x + 3) / 6
class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)
    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        #c*1*W
        x_h = self.pool_h(x)
        #c*H*1
        #C*1*h
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        #C*1*(h+w)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out = identity * a_w * a_h
        return out

class AASPP_CA(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, M=2, r=16, L=32):
        super(AASPP_CA, self).__init__()

        # Ensure in_channels is divisible by groups

        d = max(in_channels // r, L)
        self.M = M
        self.out_channels = out_channels
        self.conv = nn.ModuleList()
        self.mean = nn.AdaptiveAvgPool2d((1, 1))  # (1,1)means ouput_dim
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, 1)
        self.conv.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride, groups=1, bias=False),  # Set groups to 1
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, stride, padding=6, dilation=6, groups=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, stride, padding=12, dilation=12, groups=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, stride, padding=18, dilation=18, groups=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))

        self.global_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc1 = nn.Sequential(nn.Conv2d(out_channels, d, 1, bias=False),
                                # nn.BatchNorm2d(d),
                                 nn.ReLU(inplace=True))
        self.fc2 = nn.Conv2d(d, out_channels * M, 1, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024, 10)

        self.conv2 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv3 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1)
        self.coreAttn = CoordAtt(in_channels,out_channels)

    def forward(self, input):
        print("Input",input.shape)
        batch_size = input.size(0)
        output = []
        for i, conv in enumerate(self.conv):
            output.append(conv(input))
        U = reduce(lambda x, y: x + y, output)
        print("U",U.shape)
        s = self.global_pool(U)
        print("s",s.shape)
        z = self.fc1(s)
        a_b = self.fc2(z)
        a_b = a_b.reshape(batch_size, self.M, self.out_channels, -1)
        a_b = self.softmax(a_b)
        a_b = list(a_b.chunk(self.M, dim=1))
        a_b = list(map(lambda x: x.reshape(batch_size, self.out_channels, 1, 1),
                       a_b))
        V = list(map(lambda x, y: x * y, output,
                     a_b))

        V = reduce(lambda x, y: x + y,
                   V)
        #空间部分

        size = input.shape[2:]
        #image_features = self.mean(input)
        #image_features = self.conv1(image_features)
        #image_features = F.upsample(image_features, size=size, mode='bilinear')
        image_features = self.coreAttn(input)
        print("image_features",image_features.shape)
        V = torch.cat([image_features, V], dim=1)
        x = self.conv2(V)
       # x = self.adaptive_pool(x)
       # x = self.conv3(x)
        #V = self.global_pool(V).view(V.size(0), -1)
        #print(V.shape)
       # V = self.fc(V)
        print("Final",x.shape)
        return x