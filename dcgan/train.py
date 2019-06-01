import mxnet as mx
import mxnet.gluon as gluon
from mxnet.gluon import Trainer
from mxnet import nd, init, autograd
from mxboard import SummaryWriter

import numpy as np
import random
import time
from viz import save_images
from utils import *


# custom weights initialization called on netG and netD
def custom_init_weights(layers):
    for layer in layers:
        classname = layer.__class__.__name__
        if classname.find('Conv') != -1:
            layer.weight.set_data(mx.ndarray.random.normal(0.0,0.02,shape=layer.weight.data().shape))
        elif classname.find('BatchNorm') != -1:
            layer.gamma.set_data(mx.ndarray.random.normal(1.0, 0.02,shape=layer.gamma.data().shape))
            layer.beta.set_data(mx.ndarray.zeros(shape=layer.beta.data().shape))

def model_init(net_G, net_D, opt, ctx):
    net_G.initialize(init.Xavier(factor_type='in', magnitude=0.01), ctx=ctx)
    custom_init_weights(net_G.base)
    if opt.netG_param != '':  # load checkpoint if needed
        net_G.load_parameters(opt.netG_param)
    print(net_G)

    net_D.initialize(mx.init.Xavier(factor_type='in', magnitude=0.01), ctx=ctx)
    if opt.netD_param != '':
        net_D.load_parameters(opt.netD_param)
    print(net_D)

    # A pass forward to initialize netG, netD (because of defered initialization)
    init_x = nd.array(np.ones(shape=(opt.batchSize, opt.nz, 1, 1)), ctx=ctx)
    init_x = net_G(init_x)
    _ = net_D(init_x)

    return net_G, net_D


def train(net_G, net_D, dataloader, opt):
    '''
    Entry of Training process
    :return:
    '''
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    mx.random.seed(opt.manualSeed)

    ctx = try_gpu()
    print("ctx: ", ctx)

    # initialize netG, netD
    net_G, net_D = model_init(net_G, net_D, opt, ctx)

    # optimizer settings
    trainer_G = Trainer(net_G.collect_params(), optimizer='adam',
                        optimizer_params={'learning_rate': opt.lrG, 'beta1': opt.beta1, 'beta2': 0.999})
    trainer_D = Trainer(net_D.collect_params(), optimizer='adam',
                        optimizer_params={'learning_rate': opt.lrD, 'beta1': opt.beta1, 'beta2': 0.999})

    print("Start training ...")

    #input = mx.nd.zeros((opt.batchSize, 3, opt.imageSize, opt.imageSize))
    #noise = mx.nd.zeros((opt.batchSize, opt.nz, 1, 1))
    fixed_noise = mx.ndarray.random.normal(shape=(opt.batchSize, opt.nz, 1, 1))

    sw = SummaryWriter(logdir='./logs', flush_secs=5)

    gen_iterations = 0
    for epoch in range(opt.num_epochs):
        data_iter = iter(dataloader)
        i = 0
        while i < len(dataloader):
            start_time = time.time()
            ############################
            # (1) Update D network
            ############################

            # train the discriminator Diters times
            if gen_iterations < 25 or gen_iterations % 500 == 0:
                Diters = 100
            else:
                Diters = opt.num_iter_D
            j = 0
            while j < Diters and i < len(dataloader):
                j += 1
                # clamp parameters to a cube
                for p in net_G.collect_params():
                    param = net_D.collect_params(p)[p]
                    param.set_data(mx.nd.clip(param.data(), opt.clamp_lower, opt.clamp_upper))

                data = next(data_iter)[0]
                data = data.as_in_context(ctx)
                i += 1

                # train with real
                with autograd.record():
                    errD_real = net_D(data)

                    # train with fake
                    noise = mx.ndarray.random.normal(shape=(opt.batchSize, opt.nz, 1, 1), ctx=ctx)
                    fake = net_G(noise)
                    errD_fake = net_D(fake.detach())
                    errD = errD_real - errD_fake
                    errD.backward()
                trainer_D.step(1)

            ############################
            # (2) Update G network
            ############################
            # in case our last batch was the tail batch of the dataloader,
            # make sure we feed a full batch of noise
            noise = mx.ndarray.random.normal(shape=(opt.batchSize, opt.nz, 1, 1), ctx=ctx)
            with autograd.record():
                fake = net_G(noise)
                errG = net_D(fake)
                errG.backward()
            trainer_G.step(1)
            gen_iterations += 1

            print('[%d/%d][%d/%d][%d] Loss_D: %f Loss_G: %f Loss_D_real: %f Loss_D_fake %f,  time:[%f]'
                  % (epoch, opt.num_epochs, i, len(dataloader), gen_iterations,
                     errD.asnumpy()[0], errG.asnumpy()[0], errD_real.asnumpy()[0], errD_fake.asnumpy()[0],
                     time.time() - start_time))

            sw.add_scalar(
                tag='loss_D',
                value=-errD.asnumpy()[0],
                global_step=gen_iterations)

            if gen_iterations % 500 == 0:
                real_cpu = data * 0.5 + 0.5
                save_images(real_cpu.asnumpy().transpose(0, 2, 3, 1), '{0}/real_samples.png'.format(opt.experiment))
                fake = net_G(fixed_noise.as_in_context(ctx))
                fake = fake * 0.5 + 0.5
                save_images(fake.asnumpy().transpose(0, 2, 3, 1),
                            '{0}/fake_samples_{1}.png'.format(opt.experiment, gen_iterations))

        # do checkpointing
        net_G.save_parameters('{0}/netG_epoch_{1}.param'.format(opt.experiment, epoch))
        net_D.save_parameters('{0}/netD_epoch_{1}.param'.format(opt.experiment, epoch))
