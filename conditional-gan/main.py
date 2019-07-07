from dotmap import DotMap
from sacred_cfg import *
from pprint import pprint

from mxnet.gluon.data import DataLoader
from dataset import MNISTDataset
from transform import transform
from model import *


@ex.config
def config():
    opt = args
    print(opt)


def dot_opt(opt):
    opt = DotMap(opt)
    print(f"Experiment config opt:")
    pprint(opt)
    return opt


@ex.automain
def main(opt):
    opt = dot_opt(opt)

    # datasets
    mnist_train_dataset = MNISTDataset(train=True, transform=transform)
    assert mnist_train_dataset

    dataloader = DataLoader(mnist_train_dataset, batch_size=opt.batch_size,
                            shuffle=True, last_batch='discard',
                            pin_memory=True, num_workers=opt.num_workers)
    print("Data ready...")

    # Conditional GAN: G and D
    netG = ConditionalG(opt)

    