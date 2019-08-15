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

### 代码解读

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

机器学习问题一般被建模成一个**优化问题**，基于GAN的无监督学习也不例外。

一般地，优化就是说：构建一个模型，并为其设定一个优化目标。初始化后，通过不断调整模型参数，使得模型**越来越靠近预设的优化目标**。

### DCGAN 的优化建模

首先画出 DCGAN 架构图（如下图），可以不需要展开G 和 D的内部结构，因为在定义优化问题（定义loss function）时，G 和 D是被视为在优化问题层面的语义明确的函数，即（拟人化解释）：

> 对G而言，就是我给你一个随机向量类型的数，你给我返回一个image。
>
> 对D而言，就是我给你一个image，你给我返回一个1或者0。
>
> 至于 G 和 D本身，在优化建模的视角，（理论上假设）为一般意义上的函数即可。至于什么函数是好的，对优化求解具有良好性质的；而什么函数又是坏的，对优化求解具有很大难度的事情，则不是优化问题建模要考虑的。

![image-20190814223737276](../../../../Dropbox/Images/image-20190814223737276.png)

闲言少叙，DCGAN的loss function 和 原始GAN一样，采用**二元分类损失（或者叫 二元交叉熵损失，简称BCE）**，它基于 D 的预测输出（sigmoid归一化之后的分类概率）：
$$
prob = \frac{1}{1 + \exp(-{pred})}
\\
L = - \sum_i {label}_i * \log({prob}_i) +
(1 - {label}_i) * \log(1 - {prob}_i)
$$

* 结论

> * GAN 中的G 和 D就是模型，它们在训练中需要不断调整自身参数。
> * GAN 中的loss function（二元分类损失函数），就是优化目标

### 代码解读

在MXNet Gluon中，可以直接如下定义一个二元交叉熵损失函数算子：

```python
loss_f = loss.SigmoidBinaryCrossEntropyLoss()
```

我们直接对着代码，说明基于上图DCGAN架构的优化建模（**对什么做优化**）。以下代码，全部在 `train.py`中，可以对照着完整代码看。

* DCGAN 的优化求解过程

  GAN一个有意思的地方就是它由2个网络（G 和 D）进行对抗而不断优化的，而非简单地针对一个网络做目标明确的事情（如分类、目标检测）。为此，原始 GAN提出的一种训练方式：每一轮交替更新D 和 G的参数。这也是DCGAN优化求解的方式。

  深刻体会到：**站在各自的视角，优化各自的优化目标，更新各自的参数。**才是真正理解了“对抗”的含义。

* D 的视角

  给定一个batch的real images，在D看来，它的优化目标就是能够区分出这些 real images 和 G基于随机向量z生成的那些fake images。

  这个优化目标用形式化语言描述就是：
  $$
  \min -( \log(D(x)) + log(1 - D(G(z))))
  $$
  该公式可以分开看：对于正样本$x$，label为1，代入BCE公式，就是$log(D(x))$；对于负样本$G(z)$，label为0，代入BCE公式，就是$log(1-D(G(z)))$。把它们加起来，就是 D的优化目标。

  ```python
  ############################
  # (1) Update D network:   maximize log(D(x)) + log(1 - D(G(z)))
  ############################
  with autograd.record():
    # train with real
    real_label = nd.ones((opt.batchSize,), ctx)
    output_D_real = net_D(data)
    # print("output_D_real: {}".format(output_D_real))
    # print("real_label: {}".format(real_label))
    err_D_real = loss_f(output_D_real, real_label)
    D_x = output_D_real.mean()
  
    # train with fake
    fake_label = nd.zeros((opt.batchSize,), ctx)
    fake = net_G(noise)
    output_D_fake = net_D(fake.detach())
    err_D_fake = loss_f(output_D_fake, fake_label)
    D_G_z1 = output_D_fake.mean()
  
    err_D = err_D_real + err_D_fake
    err_D.backward()
  trainer_D.step(1)
  ```

  上面是 D 优化目标的构建和优化求解代码：

  * `# train with real` 代码块中，计算$-log(D(x))$；

  * `# train with fake` 代码块中，计算$-log(1 - D(G(z)))$。

  * `err_D_real + err_D_fake` 将两者加在一起，完成了完整的前向传播。
  * `err_D.backward()` 完成反向传播，计算D所有参数的梯度。（为什么没有G的梯度，因为这是D的优化目标，而不是G的。通过`net_D(fake.detach())`中的detach函数实现梯度脱钩）。
  * `trainer_D.step(1)` 完成一次D所有参数的更新。

  

* G的视角

  在G看来，它的优化目标就是“欺骗”D。G期望D把自己基于随机向量z生成的那些fake images 分成正类（不考虑D对 real_images的分类性能）。

  故这个优化目标用形式化语言描述只有一项，就是：
  $$
  \min - \log(D(G(z)))
  $$
  该公式的意义：对于样本$G(z)$，label为1，代入BCE公式，就是$log(D(x))$。

  可以看到对于同样的$G(z)$，G 和 D的优化目标完全相反。

  ```python
  ############################
  # (2) Update G network    maximize log(D(G(z)))
  ############################
  noise = mx.ndarray.random.normal(shape=(opt.batchSize, opt.nz, 1, 1), ctx=ctx)
  real_label = nd.ones((opt.batchSize,), ctx)
  with autograd.record():
    fake = net_G(noise)
    output_G = net_D(fake)
    err_G = loss_f(output_G, real_label)
    D_G_z2 = output_G.mean()
    err_G.backward()
  trainer_G.step(1)
  ```

  - 上面是 G 优化目标的构建和优化求解代码：

    - `with autograd.record():` 代码块中，计算$-log(D(G(z)))$

    - `err_G.backward()` 完成反向传播，计算G所有参数的梯度。
    - `trainer_G.step(1)` 完成一次G所有参数的更新。

# Experiments

### 探究 CNN 在GAN稳定训练上的作用

##### 实验说明

这是一个简单的对照实验，原始GAN训练稳定性很差。通过对比经典的基于MLP 特征提取的GAN，来探究CNN作为特征提取器，是否有效地提升GAN训练的稳定性。

##### 实验步骤

* 选择数据集

  LSUN bedroom dataset




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







