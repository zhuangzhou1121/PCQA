# Attention机制

注意力机制最早用于自然语言处理领域(NLP)，后来在计算机视觉领域(CV)也得到广泛的应用，注意力机制被引入来进行视觉信息处理。注意力机制没有严格的数学定义，例如**传统的局部图像特征提取**、**滑动窗口方法**等都可以看作一种注意力机制。在神经网络中，注意力机制通常是一个额外的神经网络，能够硬性选择输入的某些部分，或者**给输入的不同部分分配不同的权重**。注意力机制能够从大量信息中筛选出重要的信息。

近几年CV领域常用的注意力机制，包括：SE（Squeeze and Excitation）、ECA（Efficient Channel Attention）、CBAM（Convolutional Block Attention Module）、CA（Coordinate attention for efficient mobile network design）

在神经网络中引入注意力机制有很多方法，以卷积神经网络（CNN）为例，可以在空间维度增加引入attention机制(如inception网络的多尺度，让并联的卷积层有不同的权重)，也可以在通道维度(channel)增加attention机制，当然也有混合维度即同时在空间维度和通道维度增加attention机制。

空间注意力模块 (look where) 对特征图每个位置进行attention调整，(x,y)二维调整，使模型关注到值得更多关注的区域上。

通道注意力模块 (look what) 分配各个卷积通道上的资源，z轴的单维度调整。

 论文：https://arxiv.org/abs/1711.07971

## SE

目的是给特征图中不同的通道赋予不同的权重，步骤如下：

1、对特征图进行Squeeze，该步骤是通过全局平均池化把特征图从大小为(N,C,H,W)转换为(N,C,1,1)，这样就达到了全局上下文信息的融合。
2、Excitation操作，该步骤使用两个全连接层，其中第一个全连接层使用ReLU激活函数，第二个全连接层采用Sigmoid激活函数，目的是将权重中映射到(0，1)之间。值得注意的是，为了减少计算量进行降维处理，将第一个全连接的输出采用输入的1/4或者1/16。
3、通过广播机制将权重与输入特征图相乘，得到不同权重下的特征图。
![image-20240415153520240](C:\Users\86157\AppData\Roaming\Typora\typora-user-images\image-20240415153520240.png)

![image-20240415153540385](C:\Users\86157\AppData\Roaming\Typora\typora-user-images\image-20240415153540385.png)

![image-20240415153549283](C:\Users\86157\AppData\Roaming\Typora\typora-user-images\image-20240415153549283.png)

![image-20240415153555157](C:\Users\86157\AppData\Roaming\Typora\typora-user-images\image-20240415153555157.png)

代码：

    import torch
    import torch.nn as nn
    
    class Se(nn.Module):
        def __init__(self, in_channel, reduction=16):
            super(Se, self).__init__()
            self.pool = nn.AdaptiveAvgPool2d(output_size=1)
            self.fc = nn.Sequential(
                nn.Linear(in_features=in_channel, out_features=in_channel//reduction, bias=False),
                nn.ReLU(),
                nn.Linear(in_features=in_channel//reduction, out_features=in_channel, bias=False),
                nn.Sigmoid()
            )
    def forward(self,x):
        out = self.pool(x)
        out = self.fc(out.view(out.size(0),-1))
        out = out.view(x.size(0),x.size(1),1,1)
        return out*x