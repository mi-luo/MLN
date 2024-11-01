

import torch.nn as nn
import torch
#import torch.functional as F
import torch.nn.functional as F


class AASPP(nn.Module):
    def __init__(self, in_channel=512, depth=256):# - 构造函数，初始化AASPP模块。in_channel是输入通道数，depth是输出通道数。
        super(AASPP, self).__init__()
        self.mean = nn.AdaptiveAvgPool2d((1, 1))  # (1,1)means ouput_dim- 创建一个自适应平均池化层，输出维度为1x1，用于提取全局上下文信息。
        self.conv = nn.Conv2d(in_channel, depth, 1, 1)#- 创建一个1x1的卷积层，用于调整通道数。
        self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block6 = nn.Conv2d(in_channel, depth, 3, 1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channel, depth, 3, 1, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(in_channel, depth, 3, 1, padding=18, dilation=18)#- 创建四个不同扩展率（dilation rate）的空洞卷积层，用于捕获不同尺度的上下文信息。
        self.conv_1x1_output = nn.Conv2d(depth * 5, depth, 1, 1)#- 创建一个1x1的卷积层，用于融合所有特征图。
        self.global_avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(4*in_channel, depth)
        self.fcs = nn.ModuleList([])
        for i in range(4):
            self.fcs.append(nn.Linear(depth, in_channel))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):# - 定义前向传播过程。
        size = x.shape[2:]# - 获取输入特征图的空间维度。
        print("x",x.shape)
        image_features = self.mean(x)#- 应用自适应平均池化层。
        image_features = self.conv(x) #应用1x1卷积。
        #image_features = F.upsample(image_features, size=size, mode='bilinear')#- 将全局上下文特征上采样到原始特征图大小。
        image_features = F.interpolate(image_features, size=size, mode='bilinear')

        atrous_block1 = self.atrous_block1(x)
        atrous_block6 = self.atrous_block6(x)
        atrous_block12 = self.atrous_block12(x)
        atrous_block18 = self.atrous_block18(x)# - 应用不同扩展率的空洞卷积。

        fea_U = torch.cat([atrous_block1,atrous_block6,atrous_block12,atrous_block18],dim=1)
        print("feaU",fea_U.shape)
        #实现全局池化

        #fea_s = fea_U.mean(-1).mean(-1)
        fea_s = self.global_avg(fea_U)
        # fea_U = torch.sum(feam, dim=1)

        print("fea_s",fea_s.shape)  # 这行代码将打印出 fea_s 的值
        fea_z = self.fc(fea_s)
        print("fea_z",fea_z.shape)
        for i, fc in enumerate(self.fcs):
            print(i, fea_z.shape)
            vector = fc(fea_z).unsqueeze_(dim=1)
            print(i, vector.shape)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector],
                                              dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        fea_v = (fea_U * attention_vectors).sum(dim=1)
        #return fea_v

        net = self.conv_1x1_output(torch.cat([image_features, fea_v], dim=1))  # - 将所有特征图连接起来，并通过1x1卷积进行融合。

        return net

if __name__ == "__main__":
    t = torch.ones((32, 256, 24, 24))
    AASPP = AASPP(256)
    out = AASPP(t)
    print(out.shape)
