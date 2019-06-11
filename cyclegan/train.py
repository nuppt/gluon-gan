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
    init_x = nd.array(np.ones(shape=(opt.batch_size, opt.input_nc, opt.crop_size, opt.crop_size)), ctx=ctx)
    init_x = net_G(init_x)
    _ = net_DY(init_x)

    # Domain Y -> X: A pass forward to initialize net_F, net_DX (because of defered initialization)
    init_x = nd.array(np.ones(shape=(opt.batch_size, opt.output_nc, opt.crop_size, opt.crop_size)), ctx=ctx)
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
        train_step(dataloader, net_G, net_F, net_DY, net_DX,
                   trainer_G, trainer_F, trainer_DY, trainer_DX,
                   loss_GAN, loss_CC, loss_Id, opt, ctx, sw, epoch)

        # do checkpointing
        net_G.save_parameters('{0}/netG_epoch_{1}.param'.format(opt.experiment, epoch))
        net_F.save_parameters('{0}/netF_epoch_{1}.param'.format(opt.experiment, epoch))
        net_DY.save_parameters('{0}/netDY_epoch_{1}.param'.format(opt.experiment, epoch))
        net_DX.save_parameters('{0}/netDX_epoch_{1}.param'.format(opt.experiment, epoch))


def train_step(dataloader, net_G, net_F, net_DY, net_DX, trainer_G, trainer_F, trainer_DY, trainer_DX, loss_GAN, loss_CC, loss_Id, opt, ctx, sw, epoch):
    for i, (real_X, real_Y) in enumerate(dataloader):
        iter_id = epoch * len(dataloader) // opt.batch_size + i

        start_time = time.time()
        real_X = real_X.as_in_context(ctx)
        real_Y = real_Y.as_in_context(ctx)

        loss_Gs, loss_G, loss_F, loss_cycle_G, loss_cycle_F, loss_idt_G, loss_idt_F, loss_DY, loss_DX = \
            optimize_parameters(net_G, net_F, net_DY, net_DX, trainer_G, trainer_F, trainer_DY, trainer_DX,
                                loss_GAN, loss_CC, loss_Id, real_X, real_Y, opt, ctx)
        #
        # print('[{:d}/{:d}][{:d}/{:d}] loss_Gs: {:}, loss_G: {:.3f}, loss_F: {:.3f}, loss_cycle_G: {:.3f}, loss_cycle_F: {:.3f}, loss_idt_G: {:.3f}, loss_idt_F: {:.3f}, loss_DY: {:3.f}, loss_DX: {:3.f}    time:[{:f}]'.format(epoch, opt.num_epochs, i, len(dataloader),
        #       loss_Gs.asnumpy()[0], loss_G.asnumpy()[0], loss_F.asnumpy()[0], loss_cycle_G.asnumpy()[0], loss_cycle_F.asnumpy()[0], loss_idt_G.asnumpy()[0], loss_idt_F.asnumpy()[0], loss_DY.asnumpy()[0], loss_DX.asnumpy()[0],
        #       time.time() - start_time))

        print(
            '[{:d}/{:d}][{:d}/{:d}] loss_Gs: {:f}, loss_G: {:.3f}, loss_F: {:.3f}, loss_cycle_G: {:.3f}, loss_cycle_F: {:.3f}, loss_idt_G: {:.3f}, loss_idt_F: {:.3f}, loss_DY: {:.3f}, loss_DX: {:.3f}    time:[{:f}]'.format(
                epoch, opt.num_epochs, i, len(dataloader),
                loss_Gs.asnumpy()[0], loss_G.asnumpy()[0], loss_F.asnumpy()[0], loss_cycle_G.asnumpy()[0],
                loss_cycle_F.asnumpy()[0], loss_idt_G.asnumpy()[0], loss_idt_F.asnumpy()[0], loss_DY.asnumpy()[0],
                loss_DX.asnumpy()[0],
                time.time() - start_time))

        sw.add_scalar(tag='loss_DY', value=-loss_DY.asnumpy()[0], global_step=iter_id)
        sw.add_scalar(tag='loss_DX', value=-loss_DX.asnumpy()[0], global_step=iter_id)

        if (epoch * len(dataloader)/opt.batch_size + i) % 1000 == 0:
            save_images(real_X.asnumpy().transpose(0, 2, 3, 1), '{0}/real_X_samples.png'.format(opt.experiment))
            save_images(real_Y.asnumpy().transpose(0, 2, 3, 1), '{0}/real_X_samples.png'.format(opt.experiment))

            fake_Y = net_G(real_X.as_in_context(ctx))
            save_images(fake_Y.asnumpy().transpose(0, 2, 3, 1),
                        '{0}/fake_Y_samples_{1}.png'.format(opt.experiment, iter_id))

            fake_X = net_F(real_Y.as_in_context(ctx))
            save_images(fake_X.asnumpy().transpose(0, 2, 3, 1),
                        '{0}/fake_X_samples_{1}.png'.format(opt.experiment, iter_id))


def get_trainers(opt, **networks):
    """Define trainers for networks according to opt."""

    net_G = networks['net_G']
    net_F = networks['net_F']
    net_DY = networks['net_DY']
    net_DX = networks['net_DX']

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
    with autograd.record():
        # CycleGAN one pass forward
        fake_Y, rec_X, fake_X, rec_Y = forward(net_G, net_F, real_X, real_Y)   # compute fake images and reconstruction images.
        # G and F
        loss_Gs, loss_G, loss_F, loss_cycle_G, loss_cycle_F, loss_idt_G, loss_idt_F = backward_Gs(opt, net_G, net_F, net_DY, net_DX,
                                                                                                  loss_GAN, loss_CC, loss_Id,
                                                                                                  ctx, real_X, real_Y, fake_Y, fake_X, rec_X, rec_Y)           # calculate gradients for G and F
    trainer_G.step(1)       # update G's weights
    trainer_F.step(1)       # update F's weights

    with autograd.record():
        # D_Y and D_X
        loss_DY = backward_DY(opt, ctx, net_DY, loss_GAN, real_Y, fake_Y)           # calculate gradients for net_DY
        loss_DX = backward_DX(opt, ctx, net_DY, loss_GAN, real_Y, fake_Y)           # calculate graidents for net_DX
    trainer_DY.step(1, ignore_stale_grad=True)      # update net_DY's weights
    trainer_DX.step(1, ignore_stale_grad=True)      # update net_DX's weights

    return loss_Gs, loss_G, loss_F, loss_cycle_G, loss_cycle_F, loss_idt_G, loss_idt_F, loss_DY, loss_DX

def forward(net_G, net_F, real_X, real_Y):
    """Run forward pass.
        1. X -> Y -> X
        2. Y -> X -> Y
    """
    fake_Y = net_G(real_X)        # G(X)
    rec_X = net_F(fake_Y)         # F(G(X)) ~ X

    fake_X = net_F(real_Y)        # F(Y)
    rec_Y = net_G(fake_X)         # G(F(Y)) ~ Y

    return fake_Y, rec_X, fake_X, rec_Y

def backward_Gs(opt, net_G, net_F, net_DY, net_DX, loss_GAN, loss_CC, loss_Id, ctx, real_X, real_Y, fake_Y, fake_X, rec_X, rec_Y):
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
    loss_idt_G = 0.
    loss_idt_F = 0.
    if opt.lambda_identity > 0:
        # G should be identity if real_Y is fed: ||G(Y) - Y||
        idt_Y = net_G(real_Y)
        loss_idt_G = loss_Id(idt_Y, real_Y) * opt.lambda_B * opt.lambda_identity
        # F should be identity if real_X is fed: ||F(X) - X||
        idt_X = net_F(real_X)
        loss_idt_F = loss_Id(idt_X, real_X) * opt.lambda_A * opt.lambda_identity

    #######################
    # GAN loss (Generator Perspective: G and F)
    #   1. DY(fake_Y, real_label)
    #       For generator G: (real_X -> fake_Y), it would cheat DY, that fake_Y generated by G is real.
    #   2. DX(fake_X, real_label)
    #       For generator F: (real_Y -> fake_X), it would cheat DX, that fake_X generated by F is real.
    #######################
    pred_DY = net_DY(fake_Y)
    real_label = nd.ones(shape=pred_DY.shape, ctx=ctx)
    loss_G = loss_GAN(pred_DY, real_label)
    #print("loss_G: {}".format(loss_G))
    pred_DX = net_DX(fake_X)
    real_label = nd.ones(shape=pred_DX.shape, ctx=ctx)
    loss_F = loss_GAN(pred_DX, real_label)
    #print("loss_F: {}".format(loss_F))


    #######################
    # Cycle Consistent loss
    #   1. F(G(X)) ~ X
    #   2. G(F(Y)) ~ Y
    #
    #       F(G(X)) represents reconstruction of X
    #       G(F(Y)) represents reconstruction of Y
    #######################
    # Forward cycle loss || F(G(X)) - X||
    loss_cycle_G = loss_CC(rec_X, real_X) * opt.lambda_A
    # Backward cycle loss || G_A(G_B(B)) - B||
    loss_cycle_F = loss_CC(rec_Y, real_Y) * opt.lambda_B

    # combined loss and calculate gradients
    loss_Gs = loss_G + loss_F + loss_cycle_G + loss_cycle_F + loss_idt_G + loss_idt_F
    loss_Gs.backward()
    return loss_Gs, loss_G, loss_F, loss_cycle_G, loss_cycle_F, loss_idt_G, loss_idt_F

def backward_DY(opt, ctx, net_DY, loss_GAN, real_Y, fake_Y):
    loss_DY = backward_D_basic(opt, ctx, net_DY, loss_GAN, real_Y, fake_Y)
    return loss_DY

def backward_DX(opt, ctx, net_DX, loss_GAN, real_X, fake_X):
    loss_DX = backward_D_basic(opt, ctx, net_DX, loss_GAN, real_X, fake_X)
    return loss_DX

def backward_D_basic(opt, ctx, net_D, loss_GAN, real, fake):
    """Calculate GAN loss and gradient for the discriminator

    Parameters:
        net_D (network)     -- the discriminator D
        real (tensor array) -- real images
        fake (tensor array) -- images generated by a generator

    Return the discriminator loss.
    We also call loss_D.backward() to calculate the gradients.
    """
    # Real
    pred_real = net_D(real)
    real_label = nd.ones(shape=pred_real.shape, ctx=ctx)
    loss_D_real = loss_GAN(pred_real, real_label)
    # Fake
    pred_fake = net_D(fake.detach())
    fake_label = nd.zeros(shape=pred_fake.shape, ctx=ctx)
    loss_D_fake = loss_GAN(pred_fake, fake_label)

    # Combined loss and calculate gradients
    loss_D = (loss_D_real + loss_D_fake) * 0.5
    loss_D.backward()
    return loss_D
