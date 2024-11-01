import torch.nn as nn
import torch
import torch.nn.functional as F

class Incep(nn.Module):
    def __init__(self, in_channel=512, depth=256):  # - 构造函数，初始化ASPP模块。in_channel是输入通道数，depth是输出通道数。
        super(Incep, self).__init__()
        self.mean = nn.AdaptiveAvgPool2d((1, 1))  # (1,1)means ouput_dim- 创建一个自适应平均池化层，输出维度为1x1，用于提取全局上下文信息。
        self.conv = nn.Conv2d(in_channel, depth, 1, 1)  # - 创建一个1x1的卷积层，用于调整通道数。
        self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block6 = nn.Conv2d(in_channel, depth, 3, 1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channel, depth, 3, 1, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(in_channel, depth, 3, 1, padding=18,
                                        dilation=18)  # - 创建四个不同扩展率（dilation rate）的空洞卷积层，用于捕获不同尺度的上下文信息。
        #self.conv_1x1_output = nn.Conv2d(in_channel, depth, 1, 1)  # - 创建一个1x1的卷积层，用于融合所有特征图。
        self.conv1x1 = nn.Conv2d(512, 256, kernel_size=1)

    def forward(self, x):  # - 定义前向传播过程。

        atrous_block1 = self.atrous_block1(x)
        atrous_block6 = self.atrous_block6(x)
        atrous_block12 = self.atrous_block12(x)
        atrous_block18 = self.atrous_block18(x)  # - 应用不同扩展率的空洞卷积。

        #print(atrous_block1.shape,atrous_block6.shape,atrous_block12.shape,atrous_block18.shape)
        #net = self.conv_1x1_output(torch.cat([atrous_block1, atrous_block6, atrous_block12, atrous_block18],
                                        #    dim=1))  # - 将所有特征图连接起来，并通过1x1卷积进行融合
        net = torch.cat([atrous_block1, atrous_block6, atrous_block12, atrous_block18],
                                            dim=1)
        #print(net.shape)
        net = self.conv1x1(net)
        return net, atrous_block1, atrous_block6, atrous_block12, atrous_block18


class SKConv(nn.Module):
    def __init__(self, features, WH, M, G, r, stride=1, L=32):
        super(SKConv, self).__init__()
        d = max(int(features / r), L)
        self.M = M
        self.features = features
        self.conv = Incep(features, d)
        self.fc1 = nn.Linear(features*2, d)
        self.fc = nn.Linear(features, d)
        self.bn = nn.BatchNorm1d(256)
        self.bn1 = nn.BatchNorm1d(128)
        self.softmax = nn.Softmax(dim=1)
        self.mean = nn.AdaptiveAvgPool2d((1, 1))  # (1,1)means ouput_dim- 创建一个自适应平均池化层，输出维度为1x1，用于提取全局上下文信息。
        self.conv1 = nn.Conv2d(32, 256, 1, 1)  # - 创建一个1x1的卷积层，用于调整通道数。
        self.conv2 = nn.Conv2d(1024, 256, kernel_size=1)
        #self.conv_1x1_output = nn.Conv2d(d * 5, d, 1, 1)  # - 创建一个1x1的卷积层，用于融合所有特征图。

    def forward(self, x):
        fea_U, atrous_block1, atrous_block6, atrous_block12, atrous_block18 = self.conv(x)
        #   fea_s = fea_U.permute(1,0,2,3)
       # print("1",fea_s.shape)
        #   fea_s = self.conv1(fea_s).mean(-1).mean(-1)
        #  fea_s = fea_s.permute(1,0)
      #  print("2",fea_s.shape)
        fea_s = fea_U
        #

        #print(fea_U.shape)
        #fea_s = self.bn(fea_s)
        # print(fea_s.shape)
        fea_z = fea_s
        # fc atrous_block1
        #print("atrous_block1", fea_z.shape)
       # fea_z = self.conv1down(fea_z)
        #print(fea_z.shape)
        vector = fea_z

        attention_vectors = vector


        # fc atrous_block6
        #print("atrous_block6", fea_z.shape)

        vector = fea_z


        attention_vectors = torch.cat([attention_vectors, vector],
                                          dim=1)
        # fc atrous_block12
        #print("atrous_block12", fea_z.shape)
        vector = fea_z



        attention_vectors = torch.cat([attention_vectors, vector],
                                          dim=1)
        # fc atrous_block18
        #print("atrous_block18", fea_z.shape)
        vector = fea_z
        attention_vectors = torch.cat([attention_vectors, vector],
                                          dim=1)

        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = self.conv2(attention_vectors)
        #print(attention_vectors.shape)
        #print(fea_U.shape)
        fea_v = (fea_U * attention_vectors)#.sum(dim=1)


        #image_features = self.mean(x)  # - 应用自适应平均池化层。
        #print("image", x.shape)
        image_features = x # 应用1x1卷积。
        size = x.shape[2:]
        image_features = F.interpolate(image_features, size=size, mode='bilinear', align_corners=False)

        #  print("image",image_features.shape)
      #  print("fea-v", fea_v.shape)
        net = torch.cat([image_features, fea_v], dim=1)
        return net


if __name__ == "__main__":
    t = torch.ones((32, 256, 24, 24))
    sk = SKConv(256, WH=1, M=2, G=1, r=2)
    out = sk(t)
    print(out.shape)
