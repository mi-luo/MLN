import torch.nn as nn
import torch
import torch.nn.functional as F

class AASPP1(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, M=2, r=16, L=32):
        super(AASPP1, self).__init__()

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
                                 nn.BatchNorm2d(d),
                                 nn.ReLU(inplace=True))
        self.fc2 = nn.Conv2d(d, out_channels * M, 1, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, 10)

    def forward(self, input):
        batch_size = input.size(0)
        output = []
        for i, conv in enumerate(self.conv):
            output.append(conv(input))
        U = reduce(lambda x, y: x + y, output)
        s = self.global_pool(U)
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
        size = input.shape[2:]
        image_features = self.mean(input)
        image_features = self.conv1(image_features)
        image_features = F.upsample(image_features, size=size, mode='bilinear')

        V = torch.cat([image_features, V], dim=1)
        V = self.global_pool(V).view(V.size(0), -1)
        print(V.shape)
        V = self.fc(V)

        return V