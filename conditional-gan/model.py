from mxnet.gluon import nn
from mxnet import nd


class ConditionalG(nn.Block):
    """Generator in Conditional GAN."""
    def __init__(self, opt):
        super(ConditionalG, self).__init__()
        self.opt = opt
        self._build_network()

    def forward(self, latent, label):
        pass

    def _build_network(self):
        #####################
        # input layers: latent, label
        #####################
        self.latent_in_blk = nn.Sequential()
        with self.latent_in_blk.name_scope():
            self.latent_in_blk.add([nn.Dense(units=self.opt.num_hidden, in_units=self.opt.num_z, activation='relu'),
                                    nn.LeakyReLU(0.2),])

        self.label_in_blk = nn.Sequential()
        with self.label_in_blk.name_scope():
            self.label_in_blk.add([nn.Embedding(input_dim=self.opt.num_classes, output_dim=self.opt.embeding_size),
                                   nn.Dense(units=self.opt.num_hidden)])

        self.mix_in_blk = nn.Lambda(lambda x, y: nd.concatenate(x, y))

        #####################
        # backbone
        #####################
        self.backbone = nn.Sequential()
        with self.backbone.name_scope():
            self.backbone.add([nn.Conv2DTranspose(channels=128, kernel_size=(4,4), strides=(2,2), padding='same'),
                               nn.LeakyReLU(0.2)])
            self.backbone.add([nn.Conv2DTranspose(channels=128, kernel_size=(4,4), strides=(2,2), padding='same'),
                               nn.LeakyReLU(0.2)])

        self.out_blk = nn.Conv2D(channels=self.opt.nc, kernel_size=(7,7), activation='tanh', padding='same')

