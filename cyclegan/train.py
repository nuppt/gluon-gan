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
    print(net_G)

    if opt.netF_param != '':
        net_F.load_parameters(opt.netF_param)
    print(net_F)

    if opt.netDY_param != '':
        net_DY.load_parameters(opt.netDY_param)
    print(net_DY)

    if opt.netDX_param != '':
        net_DX.load_parameters(opt.netDX_param)
    print(net_DX)

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

    # # optimizer settings
    # trainer_G = Trainer(net_G.collect_params(), optimizer='adam',
    #                     optimizer_params={'learning_rate': opt.lrG, 'beta1': opt.beta1, 'beta2': 0.999})
    # trainer_D = Trainer(net_D.collect_params(), optimizer='adam',
    #                     optimizer_params={'learning_rate': opt.lrD, 'beta1': opt.beta1, 'beta2': 0.999})
    #
    # loss_f = loss.SigmoidBinaryCrossEntropyLoss()
    #
    # print("Start training ...")
    #
    # for epoch in range(opt.num_epochs):
    #     train_step(dataloader, net_G, net_D, trainer_G, trainer_D, loss_f, opt, ctx, sw, epoch)
    #
    #     # do checkpointing
    #     net_G.save_parameters('{0}/netG_epoch_{1}.param'.format(opt.experiment, epoch))
    #     net_D.save_parameters('{0}/netD_epoch_{1}.param'.format(opt.experiment, epoch))


def train_step(dataloader, net_G, net_D, trainer_G, trainer_D, loss_f, opt, ctx, sw, epoch):
    for i, (data, _) in enumerate(dataloader):
        iter_id = epoch * len(dataloader) // opt.batchSize + i

        start_time = time.time()
        data = data.as_in_context(ctx)
        noise = mx.ndarray.random.normal(shape=(opt.batchSize, opt.nz, 1, 1), ctx=ctx)

        ############################
        # (1) Update D network:   maximize log(D(x)) + log(1 - D(G(z)))
        ############################
        with autograd.record():
            # train with real
            real_label = nd.ones((opt.batchSize,), ctx)
            output_D_real = net_D(data)
            # print("output_D_real: {}".format(output_D_real))
            # print("real_label: {}".format(real_label))
            err_D_real = loss_f(output_D_real, real_label)
            D_x = output_D_real.mean()

            # train with fake
            fake_label = nd.zeros((opt.batchSize,), ctx)
            fake = net_G(noise)
            output_D_fake = net_D(fake.detach())
            err_D_fake = loss_f(output_D_fake, fake_label)
            D_G_z1 = output_D_fake.mean()

            err_D = err_D_real + err_D_fake
            err_D.backward()
        trainer_D.step(1)

        ############################
        # (2) Update G network    maximize log(D(G(z)))
        ############################
        # in case our last batch was the tail batch of the dataloader,
        # make sure we feed a full batch of noise
        noise = mx.ndarray.random.normal(shape=(opt.batchSize, opt.nz, 1, 1), ctx=ctx)
        real_label = nd.ones((opt.batchSize,), ctx)
        with autograd.record():
            fake = net_G(noise)
            output_G = net_D(fake)
            err_G = loss_f(output_G, real_label)
            D_G_z2 = output_G.mean()
            err_G.backward()
        trainer_G.step(1)

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
