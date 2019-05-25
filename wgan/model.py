from mxnet.gluon import nn


class MLP_G(nn.Block):
    def __init__(self, size_img, num_z, num_hidden, num_c):
        super(MLP_G, self).__init__()

        self.size_img = size_img
        self.num_c = num_c

        with self.name_scope():
            self.base = nn.Sequential()
            self.base.add(nn.Dense(units=num_hidden, in_units=num_z, activation='relu'))
            self.base.add(nn.Dense(units=num_hidden, activation='relu'))
            self.base.add(nn.Dense(units=num_hidden, activation='relu'))
            self.base.add(nn.Dense(units=num_c * size_img * size_img))

    def forward(self, input):
        # input = input.reshape((input.shape[0], input.shape[1]))
        input = input.reshape((input.shape[0], -1))
        output = self.base(input)
        return output.reshape((output.shape[0], self.num_c, self.size_img, self.size_img))


class MLP_D(nn.Block):
    def __init__(self, size_img, num_hidden, num_c):
        super(MLP_D, self).__init__()

        # self.size_img = size_img
        # self.num_hidden = num_hidden
        # self.num_c = num_c

        with self.name_scope():
            self.base = nn.Sequential()
            self.base.add(nn.Dense(units=num_hidden, in_units=num_c * size_img * size_img, activation='relu'))
            self.base.add(nn.Dense(units=num_hidden, activation='relu'))
            self.base.add(nn.Dense(units=num_hidden, activation='relu'))
            self.base.add(nn.Dense(units=1))

    def forward(self, input):
        input = input.reshape((input.shape[0], -1))
        output = self.base(input)
        output = output.mean(axis=0)
        return output #.reshape(1)


class DCGAN_G(nn.Block):
    def __init__(self, size_img, num_z, num_hidden, num_c, num_extra_layers=0):
        super(DCGAN_G, self).__init__()

        cngf, t_size_img = num_hidden // 2, 4
        while t_size_img != size_img:
            cngf = cngf * 2
            t_size_img = t_size_img * 2
        num_hidden = cngf

        with self.name_scope():
            self.base = nn.Sequential()

            # inpurt is Z (nz dim vector), mapping to a convNet
            self.base.add(nn.Conv2DTranspose(channels=num_hidden, in_channels=num_z,
                                             kernel_size=4, strides=1, padding=0, use_bias=False))
            self.base.add(nn.BatchNorm(in_channels=num_hidden))
            self.base.add(nn.LeakyReLU(0))

            size_conv = 4

            # 不断 DeConv，直到 feature map 到达 size_img 的一半大小
            while size_conv < size_img // 2:
                self.base.add(nn.Conv2DTranspose(channels=num_hidden // 2, in_channels=num_hidden,
                                                 kernel_size=4, strides=2, padding=1, use_bias=False))
                self.base.add(nn.BatchNorm(in_channels=num_hidden // 2))
                self.base.add(nn.LeakyReLU(0))
                num_hidden = num_hidden // 2
                size_conv *= 2

            for _ in range(num_extra_layers):
                self.base.add(nn.Conv2D(channels=num_hidden, in_channels=num_hidden,
                                        kernel_size=3, strides=1, padding=1, use_bias=False))
                self.base.add(nn.BatchNorm(in_channels=num_hidden))
                self.base.add(nn.LeakyReLU(0))

            self.base.add(nn.Conv2DTranspose(channels=num_c, in_channels=num_hidden,
                                             kernel_size=4, strides=2, padding=1, use_bias=False, activation='tanh'))

    def forward(self, input):
        output = self.base(input)
        return output


class DCGAN_D(nn.Block):
    def __init__(self, size_img, num_c, num_hidden, num_extra_layers=0):
        super(DCGAN_D, self).__init__()

        assert size_img % 16 == 0, "isize has to be a multiple of 16"

        with self.name_scope():
            self.base = nn.Sequential()
            self.base.add(nn.Conv2D(channels=num_hidden, in_channels=num_c,
                                    kernel_size=4, strides=2, padding=1, use_bias=False))
            self.base.add(nn.LeakyReLU(0.2))

            size_conv = size_img // 2

            # *** Common Conv2DNet ***
            # Extra layers
            for _ in range(num_extra_layers):
                self.base.add(nn.Conv2D(channels=num_hidden, in_channels=num_hidden,
                                        kernel_size=3, strides=1, padding=1, use_bias=False))
                self.base.add(nn.BatchNorm(in_channels=num_hidden))
                self.base.add(nn.LeakyReLU(0.2))

            while size_conv > 4:
                in_c = num_hidden
                out_c = num_hidden * 2
                self.base.add(nn.Conv2D(channels=out_c, in_channels=in_c,
                                        kernel_size=4, strides=2, padding=1, use_bias=False))
                self.base.add(nn.BatchNorm(in_channels=out_c))
                self.base.add(nn.LeakyReLU(0.2))

                num_hidden = num_hidden * 2
                size_conv = size_conv // 2

            self.base.add(nn.Conv2D(channels=1, in_channels=num_hidden,
                                    kernel_size=4, strides=1, padding=0, use_bias=False))

    def forward(self, input):
        output = self.base(input)
        output = output.mean(axis=0)
        return output.reshape(1)

