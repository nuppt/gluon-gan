from cmd_opt import parse_args
from dataset import UnpairedDataset
from model import *
from train import train

from mxnet.gluon.data.vision import transforms
from mxnet.gluon.data import DataLoader


if __name__ == "__main__":
    opt = parse_args()

    dataset = UnpairedDataset(opt)

    assert dataset
    data_loader = DataLoader(dataset, batch_size=opt.batch_size,
                            shuffle=True, last_batch='discard',
                            pin_memory=True, num_workers=opt.workers)

    print("Data ready...")

    # for i, data in enumerate(data_loader):
    #     real_x = data['x']
    #     real_y = data['y']
    #     print(real_x.shape)
    #     print(real_y.shape)
    #     break

    # CycleGAN G and D
    # corresponding naming between paper and code:
    # paper      code           description
    # G          net_G_X2Y      generator for X -> Y
    # F          net_G_Y2X      generator for Y -> X
    # D_Y        net_D_Y        discriminator on domain Y
    # D_X        net_D_X        discriminator on domain X
    net_G_X2Y = CycleGAN_G(opt.input_nc, opt.output_nc, opt.ngf, opt.norm, not opt.no_dropout, opt.netG_arch)
    net_G_Y2X = CycleGAN_G(opt.output_nc, opt.input_nc, opt.ngf, opt.norm, not opt.no_dropout, opt.netG_arch)
    #print(net_G_X2Y)

    net_D_Y = CycleGAN_D(opt.output_nc, opt.ndf, opt.n_layers_D, opt.norm)
    net_D_X = CycleGAN_D(opt.input_nc, opt.ndf, opt.n_layers_D, opt.norm)

    print("Network ready...")
    train(net_G_X2Y, net_G_Y2X, net_D_Y, net_D_X, data_loader, opt)
