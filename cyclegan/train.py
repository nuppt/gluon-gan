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


def init_networks(net_G, net_F, net_DY, net_DX, opt, ctx):
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
                layer.initialize(init.Xavier('gaussian', factor_type='avg', magnitude=opt.init_gain), force_reinit=True, ctx=ctx)
            elif opt.init_type == 'orthogonal':
                layer.initialize(init.Orthogonal(scale=opt.init_gain), force_reinit=True, ctx=ctx)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % opt.init_type)

            if hasattr(layer, 'bias') and layer.bias is not None:
                layer.initialize(init.Constant(0.), force_reinit=True, ctx=ctx)
                # layer.bias.set_data(mx.ndarray.zeros(shape=layer.bias.data().shape))

        elif classname.find('BatchNorm') != -1:
            layer.gamma.set_data(mx.ndarray.random.normal(1.0, opt.init_gain, shape=layer.gamma.data().shape))
            layer.beta.set_data(mx.ndarray.zeros(shape=layer.beta.data().shape))

    net_G.initialize(init.Xavier(factor_type='in', magnitude=0.01), ctx=ctx)
    net_G.apply(lambda x: _init_layers(x, opt, ctx))

    net_F.initialize(init.Xavier(factor_type='in', magnitude=0.01), ctx=ctx)
    net_F.apply(lambda x: _init_layers(x, opt, ctx))

    net_DY.initialize(init.Xavier(factor_type='in', magnitude=0.01), ctx=ctx)
    net_DY.apply(lambda x: _init_layers(x, opt, ctx))

    net_DX.initialize(init.Xavier(factor_type='in', magnitude=0.01), ctx=ctx)
    net_DX.apply(lambda x: _init_layers(x, opt, ctx))

    # load checkpoint if needed
    if opt.netG_param != '':
        net_G.load_parameters(opt.netG_param)
    #print(net_G)

    if opt.netF_param != '':
        net_F.load_parameters(opt.netF_param)
    #print(net_F)

    if opt.netDY_param != '':
        net_DY.load_parameters(opt.netDY_param)
    #print(net_DY)

    if opt.netDX_param != '':
        net_DX.load_parameters(opt.netDX_param)
    #print(net_DX)

    # Domain X -> Y: A pass forward to initialize net_G, net_DY (because of defered initialization)
    init_x = nd.array(np.ones(shape=(opt.batch_size, opt.input_nc, opt.image_size, opt.image_size)), ctx=ctx)
    init_x = net_G(init_x)
    _ = net_DY(init_x)

    # Domain Y -> X: A pass forward to initialize net_F, net_DX (because of defered initialization)
    init_x = nd.array(np.ones(shape=(opt.batch_size, opt.output_nc, opt.image_size, opt.image_size)), ctx=ctx)
    init_x = net_F(init_x)
    _ = net_DX(init_x)

    return net_G, net_F, net_DY, net_DX


def train(net_G, net_F, net_DY, net_DX, dataloader, opt):
    '''
    Entry of Training process
    :return:
    '''
    sw = SummaryWriter(logdir='./logs', flush_secs=5)

    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    mx.random.seed(opt.manualSeed)

    ctx = try_gpu()
    print("ctx: ", ctx)

    # initialize netG, netD
    net_G, net_F, net_DY, net_DX = init_networks(net_G, net_F, net_DY, net_DX, opt, ctx)

    # optimizer/trainer settings
    trainer_G, trainer_F, trainer_DY, trainer_DX = get_trainers(opt, net_G=net_G, net_F=net_F, net_DY=net_DY, net_DX=net_DX)

    # define loss functions
    loss_GAN, loss_CC, loss_Id = get_loss_functions(opt)

    print("Start training ...")
    for epoch in range(opt.num_epochs):
        train_step(dataloader, net_G, net_F, net_DY, net_DX, trainer_G, trainer_F, trainer_DY, trainer_DX,
                   loss_GAN, loss_CC, loss_Id, opt, ctx, sw, epoch)

        # do checkpointing
        net_G.save_parameters('{0}/netG_epoch_{1}.param'.format(opt.experiment, epoch))
        net_D.save_parameters('{0}/netD_epoch_{1}.param'.format(opt.experiment, epoch))


def train_step(dataloader, net_G, net_F, net_DY, net_DX, trainer_G, trainer_F, trainer_DY, trainer_DX, loss_GAN, loss_CC, loss_Id, opt, ctx, sw, epoch):
    for i, (real_X, real_Y) in enumerate(dataloader):
        iter_id = epoch * len(dataloader) // opt.batchSize + i

        start_time = time.time()
        real_X = real_X.as_in_context(ctx)
        real_Y = real_Y.as_in_context(ctx)

        optimize_parameters(net_G, net_F, net_DY, net_DX, trainer_G, trainer_F, trainer_DY, trainer_DX, loss_GAN, loss_CC, loss_Id, real_X, real_Y, opt, ctx)

        print('[%d/%d][%d/%d] Loss_D: %f, Loss_G: %f, Loss_D_real(D(x)): %f, D_G_z1: %f, D_G_z2: %f, time:[%f]'
              % (epoch, opt.num_epochs, i, len(dataloader),
                 err_D.asnumpy()[0], err_G.asnumpy()[0], D_x.asnumpy()[0], D_G_z1.asnumpy()[0], D_G_z2.asnumpy()[0],
                 time.time() - start_time))

        sw.add_scalar(
            tag='loss_D',
            value=-err_D.asnumpy()[0],
            global_step=iter_id)

        if (epoch * len(dataloader)/opt.batchSize + i) % 100 == 0:
            save_images(data.asnumpy().transpose(0, 2, 3, 1), '{0}/real_samples.png'.format(opt.experiment))
            fake = net_G(noise.as_in_context(ctx))
            save_images(fake.asnumpy().transpose(0, 2, 3, 1),
                        '{0}/fake_samples_{1}.png'.format(opt.experiment, iter_id))


def get_trainers(opt, **networks):
    """Define trainers for networks according to opt."""

    net_G = networks['net_G']
    net_F = networks['net_F']
    net_DY = networks['net_DY']
    net_DX= networks['net_DX']

    trainer_G = Trainer(net_G.collect_params(), optimizer='adam',
                        optimizer_params={'learning_rate': opt.lrG, 'beta1': opt.beta1, 'beta2': 0.999})
    trainer_F = Trainer(net_F.collect_params(), optimizer='adam',
                        optimizer_params={'learning_rate': opt.lrG, 'beta1': opt.beta1, 'beta2': 0.999})
    trainer_DY = Trainer(net_DY.collect_params(), optimizer='adam',
                         optimizer_params={'learning_rate': opt.lrD, 'beta1': opt.beta1, 'beta2': 0.999})
    trainer_DX = Trainer(net_DX.collect_params(), optimizer='adam',
                         optimizer_params={'learning_rate': opt.lrD, 'beta1': opt.beta1, 'beta2': 0.999})

    return trainer_G, trainer_F, trainer_DY, trainer_DX

def get_loss_functions(opt):
    """Define loss functions."""
    loss_GAN = loss.SigmoidBinaryCrossEntropyLoss()
    loss_Cycle_Consistent = loss.L1Loss()
    loss_Identity = loss.L1Loss()

    return loss_GAN, loss_Cycle_Consistent, loss_Identity


def optimize_parameters(net_G, net_F, net_DY, net_DX, trainer_G, trainer_F, trainer_DY, trainer_DX, loss_GAN, loss_CC, loss_Id, real_X, real_Y, opt, ctx):
    """Calculate losses, gradients, and update network weights; called in every training iteration"""
    # CycleGAN one pass forward
    fake_Y, rec_X, fake_X, rec_Y = forward(net_G, net_F, real_X, real_Y)   # compute fake images and reconstruction images.

    # G and F
    backward_Gs()           # calculate gradients for G and F
    trainer_G.step(1)       # update G's weights
    trainer_F.step(1)       # update F's weights

    # D_Y and D_X
    backward_DY()           # calculate gradients for net_DY
    backward_DX()           # calculate graidents for net_DX
    trainer_DY.step(1)      # update net_DY's weights
    trainer_DX.step(1)      # update net_DX's weights

def forward(net_G, net_F, real_X, real_Y):
    """Run forward pass.
        1. X -> Y -> X
        2. Y -> X -> Y
    """

    fake_Y = net_G(real_X)        # G(X)
    rec_X = net_F(fake_Y)         # F(G(X)) ~ X

    fake_X = net_F(real_Y)  # G_B(B)
    rec_Y = net_G(fake_X)   # G_A(G_B(B))
    return fake_Y, rec_X, fake_X, rec_Y

def backward_Gs(opt, net_G, net_F, net_DY, net_DX, real_X, real_Y, loss_GAN, loss_CC, loss_Id):
    """Calculate the loss for generators G and F"""
    lambda_idt = opt.lambda_identity
    lambda_A = opt.lambda_A
    lambda_B = opt.lambda_B

    loss_idt_X = 0
    loss_idt_Y = 0
    # Identity loss
    if lambda_idt > 0:
        # G_A should be identity if real_B is fed: ||G_A(B) - B||
        idt_X = net_G(real_Y)
        loss_idt_G = loss_Id(idt_X, real_Y) * lambda_B * lambda_idt
        # G_B should be identity if real_A is fed: ||G_B(A) - A||
        idt_Y = net_F(real_X)
        loss_idt_F = loss_Id(idt_Y, real_X) * lambda_A * lambda_idt

    # GAN loss DY(G(X))
    loss_G_A = loss_GAN(netD_A(fake_B), True)
    # GAN loss D_B(G_B(B))
    loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
    # Forward cycle loss || G_B(G_A(A)) - A||
    loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
    # Backward cycle loss || G_A(G_B(B)) - B||
    loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
    # combined loss and calculate gradients
    loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
    loss_G.backward()