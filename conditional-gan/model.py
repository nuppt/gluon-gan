from mxnet.gluon import nn
from mxnet import nd


class ConditionalG(nn.Block):
    """Generator in Conditional GAN."""
    def __init__(self, opt):
        super(ConditionalG, self).__init__()
        self.opt = opt
        self._build_network()

    def forward(self, latent, label):
        latent = self.latent_in_blk(latent)
        label = self.label_in_blk(label)
        x = self.mix_in_blk(latent, label)
        out = self.out_blk(x)
        return out

    def _build_network(self):
        #####################
        # input layers: latent, label
        #####################
        self.latent_in_blk = nn.Sequential()
        with self.latent_in_blk.name_scope():
            self.latent_in_blk.add(nn.Dense(units=self.opt.z_dim, in_units=self.opt.nz, activation='relu'))
            self.latent_in_blk.add(nn.LeakyReLU(0))

        self.label_in_blk = nn.Sequential()
        with self.label_in_blk.name_scope():
            # self.label_in_blk.add(nn.Embedding(input_dim=self.opt.num_classes, output_dim=self.opt.embed_dim))
            # self.label_in_blk.add(nn.Dense(units=self.opt.ngf))
            self.label_in_blk.add(nn.Dense(units=self.opt.label_dim, in_units=self.opt.num_classes, activation='relu'))
            self.label_in_blk.add(nn.LeakyReLU(0))

        self.mix_in_blk = nn.Lambda(lambda x, y: nd.concatenate([x, y], axis=1))
        self.out_blk = nn.Dense(units=self.opt.out_dim, activation='sigmoid')


class ConditionalD(nn.Block):
    """Discriminator in Conditional GAN."""
    def __init__(self, opt):
        super(ConditionalD, self).__init__()
        self.opt = opt
        self._build_network()

    def forward(self, image, label):
        img_tensor = nd.reshape(image, shape=(self.opt.batch_size, self.opt.image_size, self.opt.image_size, self.opt.nc))
        x = self.mix_in_blk(img_tensor, self.label_in_blk(label))
        x = self.backbone(x)
        out = self.out_blk(x)
        return out

    def _build_network(self):
        #####################
        # input layers: label
        #####################
        self.label_in_blk = nn.Sequential()
        with self.label_in_blk.name_scope():
            self.label_in_blk.add(nn.Embedding(input_dim=self.opt.num_classes, output_dim=self.opt.embeding_size))
            self.label_in_blk.add(nn.Dense(units=self.opt.ndf))

        self.mix_in_blk = nn.Lambda(lambda x, y: nd.concatenate(x, y))

        #####################
        # backbone
        #####################
        self.backbone = nn.Sequential()
        with self.backbone.name_scope():
            self.backbone.add(nn.Conv2DTranspose(channels=128, kernel_size=(4, 4), strides=(2, 2)))
            self.backbone.add(nn.LeakyReLU(0.2))
            self.backbone.add(nn.Conv2DTranspose(channels=128, kernel_size=(4, 4), strides=(2, 2)))
            self.backbone.add(nn.LeakyReLU(0.2))
            self.backbone.add(nn.Flatten())
            self.backbone.add(nn.Dropout(0.5))

        self.out_blk = nn.Dense(units=1, activation='sigmoid')
