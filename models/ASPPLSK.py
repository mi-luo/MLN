import torch.nn as nn
import torch
import torch.nn.functional as F
from functools import reduce

class ASPPLSK(nn.Module):
    def __init__(self, in_channel=3, depth=16):
        super(ASPPLSK, self).__init__()
        self.mean = nn.AdaptiveAvgPool2d((1, 1))  # (1,1)means ouput_dim
        self.conv = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block6 = nn.Conv2d(in_channel, depth, 3, 1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channel, depth, 3, 1, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(in_channel, depth, 3, 1, padding=18, dilation=18)
        self.conv_1x1_output = nn.Conv2d(32, 1, 1, 1)
        self.conv1x1 = nn.Conv2d(depth, 1, kernel_size=1, stride=1)  # 可更改

        # 第一个注意力分支，通过1x1卷积减少通道数
        self.convtr1 = nn.Conv2d(depth, depth // 4, 1)

        # 第二个注意力分支，通过1x1卷积减少通道数
        self.convtr6 = nn.Conv2d(depth, depth // 4, 1)
        self.convtr12 = nn.Conv2d(depth, depth // 4, 1)
        self.convtr18 = nn.Conv2d(depth, depth // 4, 1)

        # 用于压缩注意力分支输出的1x1卷积
        self.conv_squeeze = nn.Conv2d(2, 4, 7, padding=3)

        # 最终的1x1卷积，用于融合注意力和原始输入
        # 最终的1x1卷积，用于融合注意力和原始输入
        self.convFinal = nn.Conv2d(2, in_channel, 1)
        self.convTrans = nn.Conv2d(in_channel, 16, 1)

    def forward(self, x):
        print("1", x.shape)
        size = x.shape[2:]

        image_features = self.mean(x)
        print(image_features.shape)
        image_features = self.conv(image_features)
        image_features = F.upsample(image_features, size=size, mode='bilinear')

        atrous_block1 = self.atrous_block1(x)
        atrous_block6 = self.atrous_block6(x)
        atrous_block12 = self.atrous_block12(x)
        atrous_block18 = self.atrous_block18(x)
        ##SKNET步骤
        attn1 = self.convtr1(atrous_block1)
        attn6 = self.convtr6(atrous_block6)
        attn12 = self.convtr12(atrous_block12)
        attn18 = self.convtr18(atrous_block18)

        # 拼接一下
        attn = torch.cat([attn1, attn6, attn12, attn18], dim=1)
        print(attn.shape)
        # 计算平均和最大值，用于融合
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)

        # 拼接平均和最大值
        agg = torch.cat([avg_attn, max_attn], dim=1)
        # 通过1x1卷积和 sigmoid 函数计算注意力权重
        sig = self.conv_squeeze(agg).sigmoid()
        print("shaoe", agg.shape, sig.shape)
        attn = agg * sig[:, 0, :, :].unsqueeze(1) + agg * sig[:, 1, :, :].unsqueeze(1) + agg * sig[:, 2, :,
                                                                                               :].unsqueeze(
            1) + agg * sig[:, 3, :, :].unsqueeze(1)
        print("&&&&&", attn.shape)
        attn = self.convFinal(attn)
        print("2", attn.shape)
        imgf = self.convTrans(attn * x)

        print("3", imgf.shape, image_features.shape)

        testOut = torch.cat([image_features, imgf], dim=1)
        print("4", testOut.shape)
        net = self.conv_1x1_output(testOut)

        print("net", net.shape)
        return net