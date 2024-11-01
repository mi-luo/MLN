import torch.nn as nn
import torch
import torch.nn.functional as F
from functools import reduce

# SA
# CBAM中的空间注意力
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        # (特征图的大小-算子的size+2*padding)/步长+1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1*h*w
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        # 2*h*w
        x = self.conv(x)
        # 1*h*w
        return self.sigmoid(x)

class AASPP_SA(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, M=2, r=16, L=32):
        super(AASPP_SA, self).__init__()

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

        self.conv2 = nn.Conv2d(in_channels=513, out_channels=512, kernel_size=1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv3 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1)
        self.coreAttn = SpatialAttention()

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
        print("V",V.shape)
        x = self.conv2(V)
       # x = self.adaptive_pool(x)
       # x = self.conv3(x)
        #V = self.global_pool(V).view(V.size(0), -1)
        #print(V.shape)
       # V = self.fc(V)
        print("Final",x.shape)
        return x