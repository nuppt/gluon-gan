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
