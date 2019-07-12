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

class CGANTrainer:
    def __init__(self, opt, train_dataset, **networks):
        """
        :param opt:  global options
        :param train_dataset:  dataset for training GAN G and D
        :param networks:  GAN G and D
        """
        self.opt = opt
        self.ctx = try_gpus(self.opt.gpus)
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
        trainerG = Trainer(self.netG.collect_params(), optimizer='adam',
                           optimizer_params={'learning_rate': self.opt.lrG, 'beta1': self.opt.beta1, 'beta2': 0.999})
        trainerD = Trainer(self.netD.collect_params(), optimizer='adam',
                           optimizer_params={'learning_rate': self.opt.lrD, 'beta1': self.opt.beta1, 'beta2': 0.999})

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
        for epoch in range(self.opt.num_epochs):
            self._train_step(epoch)

            # do checkpoints
            self.netG.save_parameters('{0}/netG_epoch_{1}.param'.format(self.opt.experiment, epoch))
            self.netD.save_parameters('{0}/netF_epoch_{1}.param'.format(self.opt.experiment, epoch))

    def _train_step(self, epoch):
        """
        train on epoch (and do some checkpoint operations)
        :param epoch:
        :return:
        """
        for i, (real, label) in enumerate(self.dataloader):
            start_time = time.time()
            iter_id = epoch * len(self.dataloader) // self.opt.batch_size + i
            self.real_img = real.as_in_context(self.ctx)
            self.label = label.as_in_context(self.ctx)

            self._optimize_parameters()  # One step of updating parameters

    def _optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration.
        1. Cal losses
        2. Cal gradients
        3. Update parameters
        """

        ############################
        # (1) Update D network:   maximize log(D(x)) + log(1 - D(G(z)))
        #   x: real image
        #   G(z): fake image
        ############################
        with autograd.record():
            real_D_label = nd.ones((self.opt.batch_size,), self.ctx)
            fake_D_label = nd.zeros((self.opt.batch_size,), self.ctx)

            pred_D_real = self.netD(self.real_img)
            fake_img = self.netG(self.)


        # Update parameters of generators G
        self.trainer_G.step(1)
