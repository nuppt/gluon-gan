from mxnet.gluon import nn
from mxnet import nd


class CycleGAN_G(nn.Block):
    def __init__(self, input_nc, output_nc, ngf=64, norm_type='batch', use_dropout=False, netG_arch='resnet_9blocks', padding_type='reflect'):
        super(CycleGAN_G, self).__init__()

        norm_layer = get_norm_layer(norm_type=norm_type)
        use_bias = norm_layer == nn.InstanceNorm

        if netG_arch == 'resnet_9blocks':
            n_blocks = 9
        elif netG_arch == 'resnet_6blocks':
            n_blocks = 6
        else:
            raise ValueError('Unknown netG_arch.')

        self.block_c7s1_64 = nn.Sequential()
        block_c7s1_64 = [nn.ReflectionPad2D(3),
                         nn.Conv2D(channels=ngf, in_channels=input_nc, kernel_size=7, strides=1, padding=0, use_bias=use_bias),
                         norm_layer(in_channels=ngf),
                         nn.LeakyReLU(0)]
        self.block_c7s1_64.add(*block_c7s1_64)

        self.block_dk = nn.Sequential()
        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            block_dk = [nn.Conv2D(in_channels=ngf * mult, channels=ngf * mult * 2, kernel_size=3, strides=2, padding=1, use_bias=use_bias),
                        norm_layer(in_channels=ngf * mult * 2),
                        nn.LeakyReLU(0)]
            self.block_dk.add(*block_dk)

        self.block_Rk = nn.Sequential()
        mult = 2 ** n_downsampling
        for i in range(n_blocks):  # add ResNet blocks
            block_Rk = [ResidualBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                                      use_dropout=use_dropout, use_bias=use_bias)]
            self.block_Rk.add(*block_Rk)

        self.block_uk = nn.Sequential()
        n_upsampling = 2
        for i in range(n_upsampling):  # add upsampling layers
            mult = 2 ** (n_upsampling - i)
            block_uk = [nn.Conv2DTranspose(in_channels=ngf*mult, channels=ngf*mult//2, kernel_size=3, strides=2,
                                           padding=1, output_padding=1, use_bias=use_bias),
                        norm_layer(in_channels=ngf*mult//2),
                        nn.LeakyReLU(0)]
            self.block_uk.add(*block_uk)

        self.block_c7s1_3 = nn.Sequential()
        block_c7s1_3 = [nn.ReflectionPad2D(3),
                        nn.Conv2D(in_channels=ngf, channels=output_nc, kernel_size=7, padding=0),
                        nn.HybridLambda('tanh')]

        self.block_c7s1_3.add(*block_c7s1_3)

    def forward(self, x):
        x = self.block_c7s1_64(x)
        x = self.block_dk(x)
        x = self.block_Rk(x)
        x = self.block_uk(x)
        x = self.block_c7s1_3(x)
        return x


class CycleGAN_D(nn.Block):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_type='batch'):
        super(CycleGAN_D, self).__init__()

        norm_layer = get_norm_layer(norm_type=norm_type)
        use_bias = norm_layer != nn.BatchNorm

        self.block_C64 = nn.Sequential()
        block_C64 = [nn.Conv2D(in_channels=input_nc, channels=ndf, kernel_size=4, strides=2, padding=1),
                     nn.LeakyReLU(0.2)]
        self.block_C64.add(*block_C64)

        self.block_C128_256_512 = nn.Sequential()
        mult = 1
        block_C128_256_512 = []
        for n in range(1, n_layers+1):
            mult_prev = mult
            mult = min(2 ** n, 8)
            block_C128_256_512 += [
                nn.Conv2D(in_channels=ndf * mult_prev, channels=ndf * mult, kernel_size=4,
                          strides=(2 if n != n_layers else 1), padding=1, use_bias=use_bias),
                norm_layer(in_channels=ndf * mult),
                nn.LeakyReLU(0.2)
            ]
        self.block_C128_256_512.add(*block_C128_256_512)

        self.block_output = nn.Sequential()
        block_output = [nn.Conv2D(in_channels=ndf * mult, channels=1, kernel_size=4, strides=1, padding=1)]  # output 1 channel prediction map
        self.block_output.add(*block_output)

    def forward(self, x):
        x = self.block_C64(x)
        x = self.block_C128_256_512(x)
        x = self.block_output(x)
        return x


###############################################################################
#                       Auxiliary Blocks and Functions
###############################################################################
class ResidualBlock(nn.Block):
    """Define the residual block for CycleGAN Generator."""

    def __init__(self, nc, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the residual block

        A residual block is a conv block with skip connections,
        Implement conv block in __init__, and implement skip connection in forward.
        """
        super(ResidualBlock, self).__init__()

        p = 0
        self.conv_block = nn.Sequential()
        if padding_type == 'reflect':
            self.conv_block.add(nn.ReflectionPad2D(1))
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        residual_conv1 = nn.Sequential()
        with residual_conv1.name_scope():
            residual_conv1.add(nn.Conv2D(in_channels=nc, channels=nc, kernel_size=3, padding=p, use_bias=use_bias))
            residual_conv1.add(norm_layer(in_channels=nc))
            residual_conv1.add(nn.LeakyReLU(0))
        self.conv_block.add(residual_conv1)

        p = 0
        if padding_type == 'reflect':
            self.conv_block.add(nn.ReflectionPad2D(1))
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        residual_conv2 = nn.Sequential()
        with residual_conv2.name_scope():
            residual_conv2.add(nn.Conv2D(in_channels=nc, channels=nc, kernel_size=3, padding=p, use_bias=use_bias))
            residual_conv2.add(norm_layer(in_channels=nc))
        self.conv_block.add(residual_conv2)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class IdentityBlock(nn.Block):
    def forward(self, x):
        return x

def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = nn.BatchNorm
    elif norm_type == 'instance':
        norm_layer = nn.InstanceNorm
    elif norm_type == 'none':
        norm_layer = lambda x: IdentityBlock()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer