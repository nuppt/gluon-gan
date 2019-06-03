from cmd_opt import parse_args
from dataset import LSUN
from model import *
from train import train

from mxnet.gluon.data.vision import transforms
from mxnet.gluon.data import DataLoader

if __name__ == "__main__":
    opt = parse_args()

    dataset = LSUN(root=opt.data_root, classes=['bedroom_train'],
                   transform=transforms.Compose([
                       transforms.Resize(opt.imageSize, keep_ratio=True, interpolation=3),
                       transforms.CenterCrop(opt.imageSize),
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]))

    assert dataset
    data_loader = DataLoader(dataset, batch_size=opt.batchSize,
                            shuffle=True, last_batch='discard',
                            pin_memory=True, num_workers=opt.workers)

    # DCGAN G and D
    net_G = DCGAN_G(opt.imageSize, opt.nz, opt.ngf, opt.nc, opt.n_extra_layers)
    net_D = DCGAN_D(opt.imageSize, opt.nc, opt.ndf, opt.n_extra_layers)

    train(net_G, net_D, data_loader, opt)
