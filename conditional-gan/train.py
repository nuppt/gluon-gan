"""
In `train`, core process is to:
  1. define init function and init network
  2. define loss function and cal loss
  3. cal gradient and update parameters
"""

from mxnet import init, autograd
from mxnet.gluon import Trainer, loss
import random
import time

from utils import *
from viz import *
from sacred_cfg import ex

class CGANTrainer:
    def __init__(self, opt, train_dataset, **networks):
        """
        :param opt:  global options
        :param train_dataset:  dataset for training GAN G and D
        :param networks:  GAN G and D
        """
        self.opt = opt

        self.ctx = try_gpu()  #try_gpus(self.opt.gpus)
        self.dataloader = train_dataset
        self.netG = networks['netG']
        self.netD = networks['netD']

    def _init_networks(self):
        self.netG.initialize(init.Xavier(factor_type='in', magnitude=0.01), ctx=self.ctx)
        self.netD.initialize(init.Xavier(factor_type='in', magnitude=0.01), ctx=self.ctx)

        init_z = nd.array(np.ones(shape=(self.opt.batch_size, self.opt.z_dim)), ctx=self.ctx)
        init_label = nd.array(np.ones(shape=(self.opt.batch_size, 1)), ctx=self.ctx)
        fake_img = self.netG(init_z, init_label)
        _ = self.netD(fake_img, init_label)

    def _define_loss(self):
        self.loss_f = loss.SigmoidBinaryCrossEntropyLoss()

    def _define_optimizers(self):
        self.trainerG = Trainer(self.netG.collect_params(), optimizer='adam',
                                optimizer_params={'learning_rate': self.opt.lr, 'beta1': self.opt.beta1, 'beta2': self.opt.beta2})
        self.trainerD = Trainer(self.netD.collect_params(), optimizer='adam',
                                optimizer_params={'learning_rate': self.opt.lr, 'beta1': self.opt.beta1, 'beta2': self.opt.beta2})

    def train(self):
        """Entry of Training process."""
        print("Random Seed: ", self.opt.manualSeed)
        random.seed(self.opt.manualSeed)
        mx.random.seed(self.opt.manualSeed)

        # initialize netGs, netDs
        self._init_networks()

        # define loss functions
        self._define_loss()

        # optimizers
        self._define_optimizers()

        print("Start training ...")

        self.real_mask = nd.ones(shape=(self.opt.batch_size))
        self.fake_mask = nd.zeros(shape=(self.opt.batch_size))
        for epoch in range(self.opt.num_epochs):
            self._train_on_epoch(epoch)
            self._do_checkpoints(epoch)

    def _train_on_epoch(self, epoch):
        """
        train on one epoch (and do some checkpoint operations)
        :param epoch:
        :return:
        """
        for i, (real, label) in enumerate(self.dataloader):
            self.real_img = nd.transpose(real, axes=(0,3,1,2)).as_in_context(self.ctx)
            self.label = label.as_in_context(self.ctx)

            batch_start = time.time()
            self._train_on_batch()
            batch_time = time.time() - batch_start

            self._monitor_on_batch(batch_time)

    @ex.capture
    def _monitor_on_batch(self, batch_time, _log):
        _log.info(f"loss D: {self.loss_D.asnumpy()[0]:.4f}\t"
                  f"loss G: {self.loss_G.asnumpy()[0]:.4f}\t"
                  f"time: {batch_time:.2f}s")
        # print('[%d/%d][%d/%d][%d] Loss_D: %f Loss_G: %f Loss_D_real: %f Loss_D_fake %f,  time:[%f]'
        #       % (epoch, self.opt.num_epochs, i, len(dataloader), gen_iterations,
        #          errD.asnumpy()[0], errG.asnumpy()[0], errD_real.asnumpy()[0], errD_fake.asnumpy()[0],
        #          time.time() - start_time))

    def _do_checkpoints(self, epoch):
        # do checkpoints
        self.netG.save_parameters('{0}/netG_epoch_{1}.param'.format(self.opt.experiment, epoch))
        self.netD.save_parameters('{0}/netD_epoch_{1}.param'.format(self.opt.experiment, epoch))

    def _train_on_batch(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration.
        1. Forward pass: Cal predictions and losses
        2. Backward pass: Cal gradients
        3. Update parameters
        """

        ############################
        # Update D network
        #
        # From D perspective,  the goal is to:
        #     maximize log(D(real_image)) + log(1 - D(fake_image))
        ############################
        with autograd.record():
            z = nd.random.normal(shape=(self.opt.batch_size, self.opt.z_dim))
            self.fake_img = self.netG(z, self.label)
            fake_pred = self.netD(self.fake_img.detach(), self.label)  # negative samples for D
            real_pred = self.netD(self.real_img, self.label)      # positive samples for D

            loss_D_real = self.loss_f(real_pred, self.real_mask)
            loss_D_fake = self.loss_f(fake_pred, self.fake_mask)
            self.loss_D = 0.5 * (loss_D_real + loss_D_fake)

        self.loss_D.backward(retain_graph=True)
        self.trainerD.step(1)

        ############################
        # Update G network
        #
        # From G perspective,  the goal is to:
        #     maximize log(D(fake_image))
        ############################
        with autograd.record():
            self.loss_G = self.loss_f(fake_pred, self.real_mask)
        self.loss_G.backward()
        self.trainerG.step(1, ignore_stale_grad=True)
