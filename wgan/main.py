from cmd_opt import parse_args
from dataset import LSUN
from model import *
from train import train

from mxnet.gluon.data.vision import transforms
from mxnet.gluon.data import DataLoader

if __name__ == "__main__":
    opt = parse_args()

    dataset = LSUN(root=opt.dataroot, classes=['bedroom_train'],
                   transform=transforms.Compose([
                       transforms.Resize(opt.imageSize, keep_ratio=True, interpolation=3),
                       transforms.CenterCrop(opt.imageSize),
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]))

    assert dataset
    dataloader = DataLoader(dataset, batch_size=opt.batchSize,
                            shuffle=True, last_batch='discard',
                            pin_memory=True, num_workers=opt.workers)

    # Choose G and D in [MLP_G, MLP_D, DCGAN_G, DCGAN_D, DCGAN_G_NOBN, DCGAN_D_NOBN]
    if opt.mlp_G:
        net_G = MLP_G(opt.imageSize, opt.nz, opt.ngf, opt.nc)
    else:
        net_G = DCGAN_G(opt.imageSize, opt.nz, opt.ngf, opt.nc, opt.n_extra_layers)

    if opt.mlp_D:
        net_D = MLP_D(opt.imageSize, opt.ndf, opt.nc)
    else:
        net_D = DCGAN_D(opt.imageSize, opt.nc, opt.ndf, opt.n_extra_layers)

    train(net_G, net_D, dataloader, opt)
