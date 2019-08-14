# Motivation

1. CV领域的应用中，CNN 在有监督任务上已经很成熟了（意思是：面向有监督任务的优化，CNN 可以得到良好的**CV表征**）。

   而 paper作者尝试基于GAN架构，将CNN引入GAN架构，得到DCGAN。并且基于GAN的优化任务（没有标签，即无监督学习），同样可以得到很好的CV表征。

2. 基于CNN架构，通过对 CV表征的研究（层的可视化、特征提取子模块复用的评估等），来挖掘CV领域的 GAN 的稳定训练技巧。

可以说，DCGAN 初看是GAN arch上的创新（引入CNN到GAN架构），而背后的动机则是利用GAN对 CV表征学习的探索（无监督学习）。

# Model Arch

![image-20190813211220642](../../../../Dropbox/Images/image-20190813211220642.png)

上图是paper中展示 generator 的一幅图，鉴于 DCGAN太过出名，大部分对GAN有了解的同学都很熟悉。下面就简要地概括一下，并对应到复现的代码段。

> 输入是一个 dim=100 的随机向量 $z$，通过几个逐级 deconv blocks之后，得到 $G(z)$。
>
> paper中没有给出 discriminator 的架构图示，普通的CNN（逐级下采样堆叠的 conv blocks构成）即可胜任。

### Code View

模型架构的核心代码均在：`model.py` 中，实现了两个类：`DCGAN_G` 和 `DCGAN_D`。

MXNet-Gluon中，实现一个network的模式是：继承`nn.Block`，并实现 `__init__` 和 `forward`。在`__init__`中，定义用于构建network的各个sub-blocks；在`forward`中，将这些sub-blocks “串起来”。

##### Generator

```python
class DCGAN_G(nn.Block):
    def __init__(self, size_img, num_z, num_hidden, num_c, num_extra_layers=0):
        super(DCGAN_G, self).__init__()
        ...
        self.base.add(nn.Conv2DTranspose(channels=num_c, in_channels=num_hidden, kernel_size=4, strides=2, padding=1, use_bias=False, activation='tanh'))
        
    def forward(self, input):
        output = self.base(input)
        return output
```

上面`forward`中的`self.base` 就是完整的 generator 网络图，输入是 随机向量$z$，输出是生成的图像（近似真实世界的高维表征）。

（写这个文档时感觉）这里代码风格不是很好，没有严格按`__init__` 和 `forward`的分工来实现 network，不太利于训练时的调试，因为训练时`forward`函数执行前向传播计算，将网络分割成一个个细粒度的子模块，可以方便查看子模块的输入输出（中间结果）。而这里直接在`__init__`里将完整的网络结构不加分割地定义为`self.base`，训练时我们只能看到网络的输入和输出，看不到网络内部层的输入输出。



##### Discriminator

```python
class DCGAN_D(nn.Block):
    def __init__(self, size_img, num_c, num_hidden, num_extra_layers=0):
        super(DCGAN_D, self).__init__()
        ...
    def forward(self, input):
        output = self.base(input)
        return nd.squeeze(output.reshape(-1, 1))
```

discriminator的实现也有generator的那个“小”问题。后面再回来慢慢完善吧。

# Optimization Modeling







# Experiments

### 探究 CNN 在GAN稳定训练上的作用

##### 实验说明

这是一个简单的对照实验，原始GAN训练稳定性很差。通过对比经典的基于MLP 特征提取的GAN，来探究CNN作为特征提取器，是否有效地提升GAN训练的稳定性。

##### 实验步骤

* 选择数据集

  LSUN bedroom dataset

* 执行训练过程

  训练过程



##### 实验结果



##### 实验结论和总结





### 探究 DCGAN 无监督学习到的表征效果

##### 实验说明

本质上，这是对上一个实验的补充解释。整篇paper所研究的核心问题：在GAN架构下，面向无监督学习的优化，CNN是否能学到具有一般泛化性质的CV表征。

如果学到了类似有监督学习那样的CV表征，自然就解释了“引入CNN为什么会对GAN训练稳定性提升”这个实验问题。（故我把这个实验看做是对上一个实验为什么有效进行解释的原因）

如何判定DCGAN学到了一般泛化的CV表征，也就是本实验要做的实验内容：

1. 将训练得到的GAN中的Discriminator的中间层可视化出来
2. 将Discriminator去掉最后的classifier层，保留其余层作为feature extractor，和现有的其他无监督模型对比，接同样的linear model做分类任务，看分类效果。



##### 实验步骤

* 数据集

  paper中选了两个：CIFAR-10 和 SVHN，这里我就选择 CIFAR-10 进行实验了。

  

* 



##### 实验结果



##### 实验结论和总结







