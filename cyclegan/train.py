import mxnet as mx
import mxnet.gluon as gluon
from mxnet.gluon import Trainer, loss
from mxnet import nd, init, autograd
from mxboard import SummaryWriter

import numpy as np
import random
import time
from viz import save_images
from utils import *


class CycleGANTrainer:
    def __init__(self, opt, dataloader, **networks):
        self.opt = opt
        self.ctx = try_gpu(self.opt.gpu_id)
        self.data_loader = dataloader

        self.net_G = networks['net_G']
        self.net_F = networks['net_F']
        self.net_DY = networks['net_DY']
        self.net_DX = networks['net_DX']

        self.sw = SummaryWriter(logdir='./logs', flush_secs=5)

    def train(self):
        """Entry of Training process."""
        print("Random Seed: ", self.opt.manualSeed)
        random.seed(self.opt.manualSeed)
        mx.random.seed(self.opt.manualSeed)

        # initialize netGs, netDs
        self._init_networks()

        # optimizers
        self._define_optimizers()

        # define loss functions
        self._define_loss_functions(gan_mode='lsgan')

        print("Start training ...")
        for epoch in range(self.opt.num_epochs):
            self._train_step(epoch)

            # do checkpoints
            self.net_G.save_parameters('{0}/netG_epoch_{1}.param'.format(self.opt.experiment, epoch))
            self.net_F.save_parameters('{0}/netF_epoch_{1}.param'.format(self.opt.experiment, epoch))
            self.net_DY.save_parameters('{0}/netDY_epoch_{1}.param'.format(self.opt.experiment, epoch))
            self.net_DX.save_parameters('{0}/netDX_epoch_{1}.param'.format(self.opt.experiment, epoch))

    def _train_step(self, epoch):
        for i, (real_X, real_Y) in enumerate(self.data_loader):
            start_time = time.time()
            iter_id = epoch * len(self.data_loader) // self.opt.batch_size + i
            self.real_X = real_X.as_in_context(self.ctx)
            self.real_Y = real_Y.as_in_context(self.ctx)

            self._optimize_parameters()  # One step of updating parameters

            print('[{:d}/{:d}][{:d}/{:d}] loss_Gs: {:f}, loss_G: {:.3f}, loss_F: {:.3f}, loss_cycle_G: {:.3f}, loss_cycle_F: {:.3f}, loss_idt_G: {:.3f}, loss_idt_F: {:.3f}, loss_DY: {:.3f}, loss_DX: {:.3f}    time:[{:f}]'.format(
                    epoch, self.opt.num_epochs, i, len(self.data_loader),
                    self.loss_Gs.asnumpy()[0], self.loss_G.asnumpy()[0], self.loss_F.asnumpy()[0],
                    self.loss_cycle_G.asnumpy()[0], self.loss_cycle_F.asnumpy()[0], self.loss_idt_G.asnumpy()[0],
                    self.loss_idt_F.asnumpy()[0], self.loss_DY.asnumpy()[0], self.loss_DX.asnumpy()[0],
                    time.time() - start_time))

            self.sw.add_scalar(tag='loss_DY', value=-self.loss_DY.asnumpy()[0], global_step=iter_id)
            self.sw.add_scalar(tag='loss_DX', value=-self.loss_DX.asnumpy()[0], global_step=iter_id)

            if (epoch * len(self.data_loader) / self.opt.batch_size + i) % 1000 == 0:
                save_images(real_X.asnumpy().transpose(0, 2, 3, 1), '{0}/real_X_samples.png'.format(self.opt.experiment))
                save_images(real_Y.asnumpy().transpose(0, 2, 3, 1), '{0}/real_Y_samples.png'.format(self.opt.experiment))
                save_images(self.fake_Y.asnumpy().transpose(0, 2, 3, 1),
                            '{0}/fake_Y_samples_{1}.png'.format(self.opt.experiment, iter_id))
                save_images(self.fake_X.asnumpy().transpose(0, 2, 3, 1),
                            '{0}/fake_X_samples_{1}.png'.format(self.opt.experiment, iter_id))

    def _init_networks(self):
        """Initialize CycleGAN G(s) and D(s)

            Ensure:
                1. network parameters
                    1.1 General init
                        [For example]
                        net_G.initialize(init.Xavier(factor_type='in', magnitude=0.01), ctx=ctx)
                    1.2 Fine-tune some specific layers
                        [For example]
                        in '_init_layers' function

                2. network locations   ctx=(cpu, gpu(i))

            :return: CycleGAN networks with initialization
            """

        def _init_layers(layer, opt, ctx):
            """Custom weights initialization on single layer"""
            classname = layer.__class__.__name__
            if hasattr(layer, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if opt.init_type == 'normal':
                    layer.weight.set_data(mx.ndarray.random.normal(0, opt.init_gain, shape=layer.weight.data().shape))
                elif opt.init_type == 'xavier':
                    layer.initialize(init.Xavier('gaussian', factor_type='avg', magnitude=opt.init_gain),
                                     force_reinit=True, ctx=ctx)
                elif opt.init_type == 'orthogonal':
                    layer.initialize(init.Orthogonal(scale=opt.init_gain), force_reinit=True, ctx=ctx)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % opt.init_type)

                if hasattr(layer, 'bias') and layer.bias is not None:
                    layer.bias.initialize(init.Constant(0.), force_reinit=True, ctx=ctx)
                    # layer.bias.set_data(mx.ndarray.zeros(shape=layer.bias.data().shape))

            elif classname.find('BatchNorm') != -1:
                layer.gamma.set_data(mx.ndarray.random.normal(1.0, opt.init_gain, shape=layer.gamma.data().shape))
                layer.beta.set_data(mx.ndarray.zeros(shape=layer.beta.data().shape))

        self.net_G.initialize(init.Xavier(factor_type='in', magnitude=0.01), ctx=self.ctx)
        self.net_G.apply(lambda x: _init_layers(x, self.opt, self.ctx))

        self.net_F.initialize(init.Xavier(factor_type='in', magnitude=0.01), ctx=self.ctx)
        self.net_F.apply(lambda x: _init_layers(x, self.opt, self.ctx))

        self.net_DY.initialize(init.Xavier(factor_type='in', magnitude=0.01), ctx=self.ctx)
        self.net_DY.apply(lambda x: _init_layers(x, self.opt, self.ctx))

        self.net_DX.initialize(init.Xavier(factor_type='in', magnitude=0.01), ctx=self.ctx)
        self.net_DX.apply(lambda x: _init_layers(x, self.opt, self.ctx))

        # load checkpoint if needed
        if self.opt.netG_param != '':
            self.net_G.load_parameters(self.opt.netG_param)
        # print(net_G)

        if self.opt.netF_param != '':
            self.net_F.load_parameters(self.opt.netF_param)
        # print(net_F)

        if self.opt.netDY_param != '':
            self.net_DY.load_parameters(self.opt.netDY_param)
        # print(net_DY)

        if self.opt.netDX_param != '':
            self.net_DX.load_parameters(self.opt.netDX_param)
        # print(net_DX)

        # Domain X -> Y: A pass forward to initialize net_G, net_DY (because of defered initialization)
        init_x = nd.array(np.ones(shape=(self.opt.batch_size, self.opt.input_nc, self.opt.crop_size, self.opt.crop_size)),
                          ctx=self.ctx)
        init_x = self.net_G(init_x)
        _ = self.net_DY(init_x)

        # Domain Y -> X: A pass forward to initialize net_F, net_DX (because of defered initialization)
        init_x = nd.array(np.ones(shape=(self.opt.batch_size, self.opt.output_nc, self.opt.crop_size, self.opt.crop_size)),
                          ctx=self.ctx)
        init_x = self.net_F(init_x)
        _ = self.net_DX(init_x)

    def _define_optimizers(self):
        """Define trainers for networks according to opt."""

        self.trainer_G = Trainer(self.net_G.collect_params(), optimizer='adam',
                                 optimizer_params={'learning_rate': self.opt.lrG, 'beta1': self.opt.beta1, 'beta2': 0.999})
        self.trainer_F = Trainer(self.net_F.collect_params(), optimizer='adam',
                                 optimizer_params={'learning_rate': self.opt.lrG, 'beta1': self.opt.beta1, 'beta2': 0.999})
        self.trainer_DY = Trainer(self.net_DY.collect_params(), optimizer='adam',
                                  optimizer_params={'learning_rate': self.opt.lrD, 'beta1': self.opt.beta1, 'beta2': 0.999})
        self.trainer_DX = Trainer(self.net_DX.collect_params(), optimizer='adam',
                                  optimizer_params={'learning_rate': self.opt.lrD, 'beta1': self.opt.beta1, 'beta2': 0.999})

    def _define_loss_functions(self, gan_mode):
        """Define loss functions.
            1. loss_GAN
            2. loss_Cycle_Consistent
            3. loss_Identity
        """
        if gan_mode == 'lsgan':
            self.loss_GAN = loss.L2Loss()
        elif gan_mode == 'vanilla':
            self.loss_GAN = loss.SigmoidBinaryCrossEntropyLoss()
        self.loss_CC = loss.L1Loss()
        self.loss_Id = loss.L1Loss()

    def _optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration

            Step 1. Forward pass
                Calculate G(x), F(G(x)), F(y), G(F(y)) and losses for Gs (G & F)

            Step 2. Backward pass of Gs
                Calculate gradients for Gs and update parameters of Gs

            Step 3. Backward pass of DY and DX
                Calculate gradients for Ds and update parameters of Ds
        """
        with autograd.record():
            # Step 1. Forward pass
            #   compute fake images and reconstruction images
            self._forward_model()
            self._forward_G_losses()
            #   calculate gradients for G and F
            self.loss_Gs.backward()
        # Update parameters of generators (G and F)
        self.trainer_G.step(1)
        self.trainer_F.step(1)

        with autograd.record():
            # D_Y and D_X
            self._forward_D_losses()
            self.loss_DY.backward()
            self.loss_DX.backward()
        self.trainer_DY.step(1, ignore_stale_grad=True)  # update net_DY's weights
        self.trainer_DX.step(1, ignore_stale_grad=True)  # update net_DX's weights

    def _forward_model(self):
        """Run forward pass.
            1. X -> Y -> X
            2. Y -> X -> Y
        """
        self.fake_Y = self.net_G(self.real_X)  # G(X)
        self.rec_X = self.net_F(self.fake_Y)  # F(G(X)) ~ X

        self.fake_X = self.net_F(self.real_Y)  # F(Y)
        self.rec_Y = self.net_G(self.fake_X)  # G(F(Y)) ~ Y

    def _forward_G_losses(self):
        """Calculate the loss for generators G and F
            1. Identity loss
            2. GAN loss
            3. Cycle Consistent loss
        """

        #######################
        # Identity loss
        #   identity means that:
        #       1. G should modify X, but not modify Y. That means G(Y) is nearly same to Y.
        #       2. F should modify Y, but not modify X. That means F(X) is nearly same to X.
        #######################
        self.loss_idt_G = 0.
        self.loss_idt_F = 0.
        if self.opt.lambda_identity > 0:
            # G should be identity if real_Y is fed: ||G(Y) - Y||
            idt_Y = self.net_G(self.real_Y)
            self.loss_idt_G = self.loss_Id(idt_Y, self.real_Y) * self.opt.lambda_B * self.opt.lambda_identity
            # F should be identity if real_X is fed: ||F(X) - X||
            idt_X = self.net_F(self.real_X)
            self.loss_idt_F = self.loss_Id(idt_X, self.real_X) * self.opt.lambda_A * self.opt.lambda_identity

        #######################
        # GAN loss (Generator Perspective: G and F)
        #   1. DY(fake_Y, real_label)
        #       For generator G: (real_X -> fake_Y), it would cheat DY, that fake_Y generated by G is real.
        #   2. DX(fake_X, real_label)
        #       For generator F: (real_Y -> fake_X), it would cheat DX, that fake_X generated by F is real.
        #######################
        pred_DY = self.net_DY(self.fake_Y)
        real_label = nd.ones(shape=pred_DY.shape, ctx=self.ctx)
        self.loss_G = self.loss_GAN(pred_DY, real_label)
        # print("loss_G: {}".format(loss_G))
        pred_DX = self.net_DX(self.fake_X)
        real_label = nd.ones(shape=pred_DX.shape, ctx=self.ctx)
        self.loss_F = self.loss_GAN(pred_DX, real_label)
        # print("loss_F: {}".format(loss_F))

        #######################
        # Cycle Consistent loss (key idea)
        #   1. F(G(X)) ~ X
        #   2. G(F(Y)) ~ Y
        #
        #       F(G(X)) represents reconstruction of X
        #       G(F(Y)) represents reconstruction of Y
        #######################
        # Forward cycle loss || F(G(X)) - X||
        self.loss_cycle_G = self.loss_CC(self.rec_X, self.real_X) * self.opt.lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_F = self.loss_CC(self.rec_Y, self.real_Y) * self.opt.lambda_B

        # combined loss and calculate gradients
        self.loss_Gs = self.loss_G + self.loss_F + self.loss_cycle_G + self.loss_cycle_F + self.loss_idt_G + self.loss_idt_F

    def _forward_D_losses(self):
        """Calculate GAN loss for Ds"""
        self.loss_DY = self._forward_D_base(self.net_DY, self.real_Y, self.fake_Y)
        self.loss_DX = self._forward_D_base(self.net_DX, self.real_X, self.fake_X)

    def _forward_D_base(self, net_D, real, fake):
        """Calculate GAN loss

        Parameters:
            net_D (network)     -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        """
        # Real
        pred_real = net_D(real)
        real_label = nd.ones(shape=pred_real.shape, ctx=self.ctx)
        loss_D_real = self.loss_GAN(pred_real, real_label)
        # Fake
        pred_fake = net_D(fake.detach())
        fake_label = nd.zeros(shape=pred_fake.shape, ctx=self.ctx)
        loss_D_fake = self.loss_GAN(pred_fake, fake_label)

        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        return loss_D
