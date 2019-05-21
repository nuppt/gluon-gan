import mxnet as mx
import mxnet.gluon as gluon
from mxnet.gluon import Trainer
from mxnet import nd, init, autograd
from mxboard import SummaryWriter

import numpy as np
from viz import save_images

# custom weights initialization called on netG and netD
def custom_init_weights(layers):
    for layer in layers:
        classname = layer.__class__.__name__
        if classname.find('Conv') != -1:
            layer.weight.set_data(mx.ndarray.random.normal(0.0,0.02,shape=layer.weight.data().shape))
        elif classname.find('BatchNorm') != -1:
            layer.gamma.set_data(mx.ndarray.random.normal(1.0, 0.02,shape=layer.gamma.data().shape))
            layer.beta.set_data(mx.ndarray.zeros(shape=layer.beta.data().shape))


def try_gpu():
    """If GPU is available, return mx.gpu(0); else return mx.cpu()"""
    try:
        ctx = mx.gpu()
        _ = nd.array([0], ctx=ctx)
    except:
        ctx = mx.cpu()
    return ctx


def try_all_gpus():
    """Return all available GPUs, or [mx.gpu()] if there is no GPU"""
    ctx_list = []
    try:
        for i in range(16):
            ctx = mx.gpu(i)
            _ = nd.array([0], ctx=ctx)
            ctx_list.append(ctx)
    except:
        pass


def train(netG, netD, dataloader, opt):
    '''
    Entry of Training process
    :return:
    '''
    ctx = try_gpu()

    # initialize netG, netD
    netG.initialize(init.Xavier(factor_type='in',magnitude=0.01), ctx=ctx)
    custom_init_weights(netG.base)
    if opt.netG != '': # load checkpoint if needed
        netG.load_parameters(opt.netG)
    print(netG)

    netD.initialize(mx.init.Xavier(factor_type='in', magnitude=0.01), ctx=ctx)
    if opt.netD != '':
        netD.load_parameters(opt.netD)
    print(netD)

    # A pass forward to initialize netG, netD (because of defered initialization)
    init_x = nd.array(np.ones(shape=(opt.batchSize, opt.nz)), ctx=ctx)  # batchsize=8, nz=100
    init_x = netG(init_x)
    _ = netD(init_x)

    # optimizer settings
    trainer_G = Trainer(netG.collect_params(), optimizer='adam',
                        optimizer_params={'learning_rate': opt.lrG, 'beta1': opt.beta1, 'beta2': 0.999})
    trainer_D = Trainer(netD.collect_params(), optimizer='adam',
                        optimizer_params={'learning_rate': opt.lrD, 'beta1': opt.beta1, 'beta2': 0.999})

    print("Start training ...")

    #input = mx.nd.zeros((opt.batchSize, 3, opt.imageSize, opt.imageSize))
    #noise = mx.nd.zeros((opt.batchSize, opt.nz, 1, 1))
    fixed_noise = mx.ndarray.random.normal(shape=(opt.batchSize, opt.nz, 1, 1))

    sw = SummaryWriter(logdir='./logs', flush_secs=5)

    gen_iterations = 0
    for epoch in range(opt.num_iter):
        data_iter = iter(dataloader)
        i = 0
        while i < len(dataloader):
            ############################
            # (1) Update D network
            ###########################

            # train the discriminator Diters times
            if gen_iterations < 25 or gen_iterations % 500 == 0:
                Diters = 100
            else:
                Diters = opt.Diters
            j = 0
            while j < Diters and i < len(dataloader):
                j += 1
                # clamp parameters to a cube
                for p in netD.collect_params():
                    param = netD.collect_params(p)[p]
                    param.set_data(mx.nd.clip(param.data(), opt.clamp_lower, opt.clamp_upper))

                data = next(data_iter)[0]
                data = data.as_in_context(ctx)
                i += 1

                # train with real
                batch_size = data.shape[0]

                with autograd.record():
                    errD_real = netD(data)

                    # train with fake
                    noise = mx.ndarray.random.normal(shape=(opt.batchSize, opt.nz, 1, 1), ctx=ctx)
                    fake = netG(noise)
                    errD_fake = netD(fake.detach())
                    errD = errD_real - errD_fake
                    #errD = mx.gluon.loss.L1Loss()(errD_real, errD_fake)
                    errD.backward()
                trainer_D.step(1)

            ############################
            # (2) Update G network
            ###########################
            # in case our last batch was the tail batch of the dataloader,
            # make sure we feed a full batch of noise
            noise = mx.ndarray.random.normal(shape=(opt.batchSize, opt.nz, 1, 1), ctx=ctx)
            with autograd.record():
                fake = netG(noise)
                errG = netD(fake)
                errG.backward()
            trainer_G.step(1)
            gen_iterations += 1

            print('[%d/%d][%d/%d][%d] Loss_D: %f Loss_G: %f Loss_D_real: %f Loss_D_fake %f'
                  % (epoch, opt.num_iter, i, len(dataloader), gen_iterations,
                     errD.asnumpy()[0], errG.asnumpy()[0], errD_real.asnumpy()[0], errD_fake.asnumpy()[0]))

            sw.add_scalar(
                tag='loss_D',
                value=-errD.asnumpy()[0],
                global_step=gen_iterations)

            if gen_iterations % 500 == 0:
                real_cpu = data * 0.5 + 0.5
                save_images(real_cpu.asnumpy().transpose(0, 2, 3, 1), '{0}/real_samples.png'.format(opt.experiment))
                fake = netG(fixed_noise.as_in_context(ctx))
                fake = fake * 0.5 + 0.5
                save_images(fake.asnumpy().transpose(0, 2, 3, 1),
                            '{0}/fake_samples_{1}.png'.format(opt.experiment, gen_iterations))

        # do checkpointing
        netG.save_parameters('{0}/netG_epoch_{1}.param'.format(opt.experiment, epoch))
        netD.save_parameters('{0}/netD_epoch_{1}.param'.format(opt.experiment, epoch))
