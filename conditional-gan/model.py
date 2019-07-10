from mxnet.gluon import nn
from mxnet import nd


class ConditionalG(nn.Block):
    """Generator in Conditional GAN."""
    def __init__(self, opt):
        super(ConditionalG, self).__init__()
        self.opt = opt
        self._build_network()

    def forward(self, latent, label):
        label = self.label_input_block(label)
        input = self.com_input_block(latent, label)
        img = self.net(input)
        print(img)
        return img

    def _build_network(self):
        #####################
        # input layers
        #####################
        z_dim = self.opt.z_dim
        num_classes = self.opt.num_classes

        # Label embedding:
        # ----------------
        # Turns labels into dense vectors of size z_dim
        # Produces 3D tensor with shape (batch_size, 1, z_dim)
        self.label_input_block = nn.Sequential()
        self.label_input_block.add(nn.Embedding(num_classes, z_dim))

        # Flatten the embedding 3D tensor into 2D tensor with shape (batch_size, z_dim)
        self.label_input_block.add(nn.Flatten())

        # Element-wise product of the vectors z and the label embeddings
        self.com_input_block = nn.Lambda(lambda x, y: x + y)

        #####################
        # backbone
        #####################
        self.net = nn.Sequential()

        # Reshape input into 7x7x256 tensor via a fully connected layer
        self.net.add(nn.Dense(units=256 * 7 * 7, in_units=z_dim))
        self.net.add(nn.Lambda(lambda x:  nd.reshape(x, shape=(-1, 7, 7, 256))))

        # Transposed convolution layer, from 7x7x256 into 14x14x128 tensor
        self.net.add(nn.Conv2DTranspose(128, kernel_size=3, strides=2, padding=(1,1)))

        # Batch normalization
        self.net.add(nn.BatchNorm())

        # Leaky ReLU activation
        self.net.add(nn.LeakyReLU(alpha=0.01))

        # Transposed convolution layer, from 14x14x128 to 14x14x64 tensor
        self.net.add(nn.Conv2DTranspose(64, kernel_size=3, strides=1, padding=(1,1)))

        # Batch normalization
        self.net.add(nn.BatchNorm())

        # Leaky ReLU activation
        self.net.add(nn.LeakyReLU(alpha=0.01))

        # Transposed convolution layer, from 14x14x64 to 28x28x1 tensor
        self.net.add(nn.Conv2DTranspose(1, kernel_size=3, strides=2, padding=(1,1)))

        # Output layer with tanh activation
        self.net.add(nn.Activation('tanh'))


class ConditionalD(nn.Block):
    """Discriminator in Conditional GAN."""
    def __init__(self, opt):
        super(ConditionalD, self).__init__()
        self.opt = opt
        self._build_network()

    def forward(self, image, label):
        pass

    def _build_network(self):
        pass