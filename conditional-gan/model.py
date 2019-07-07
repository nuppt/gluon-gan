from mxnet.gluon import nn
from mxnet import nd


class ConditionalG(nn.Block):
    """Generator in Conditional GAN."""
    def __init__(self, opt):
        super(ConditionalG, self).__init__()
        self.opt = opt
        self._build_network()

    def forward(self, x):
        return self.net(x)

    def _build_network(self):
        with self.name_scope():
            self.net = nn.Sequential()
            self.net.add(nn.Dense(units=self.opt.num_hidden, in_units=self.opt.num_z, activation='relu'))
            self.net.add(nn.Dense(units=self.opt.num_c * self.opt.size_img * self.opt.size_img))
