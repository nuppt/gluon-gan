import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='ae_photos | apple2orange | summer2winter_yosemite | '
                                                         'horse2zebra | monet2photo | cezanne2photo | ukiyoe2photo '
                                                         '| vangogh2photo | maps | cityscapes | facades | iphone2dslr_flower ')
    parser.add_argument('--data_root', required=True, help='path to dataset')
    parser.add_argument('--max_dataset_size', type=int, default=float("inf"),
                        help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
    parser.add_argument('--serial_batches', action='store_true',
                        help='if true, takes images in order to make batches, otherwise takes them randomly')
    parser.add_argument('--direction', type=str, default='X2Y', help='X2Y or Y2X')
    parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
    parser.add_argument('--manualSeed', type=int, default=10, help='seed number for result reproduction')
    parser.add_argument('--preprocess', type=str, default='resize_and_crop',
                        help='scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]')
    parser.add_argument('--no_flip', action='store_true',
                        help='if specified, do not flip the images for data augmentation')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
    parser.add_argument('--image_size', type=int, default=288, help='scale images to this size')
    parser.add_argument('--crop_size', type=int, default=256, help='then crop to this size')
    parser.add_argument('--input_nc', type=int, default=3,
                        help='# of input image channels: 3 for RGB and 1 for grayscale')
    parser.add_argument('--output_nc', type=int, default=3,
                        help='# of output image channels: 3 for RGB and 1 for grayscale')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=25, help='number of epochs to train for')
    parser.add_argument('--lrD', type=float, default=0.0002, help='learning rate for Critic, default=0.0002')
    parser.add_argument('--lrG', type=float, default=0.0002, help='learning rate for Generator, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--netG_arch', type=str, default='resnet_9blocks', help='resnet_9blocks | resnet_6blocks')
    parser.add_argument('--netD_arch', type=str, default='n_layers', help='basic | n_layers')
    parser.add_argument('--n_layers_D', type=int, default=3, help='only used if netD==n_layers')
    parser.add_argument('--norm', type=str, default='instance',
                        help='instance normalization or batch normalization [instance | batch | none]')
    parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
    parser.add_argument('--netG_param', default='', help="path to netG (to continue training)")
    parser.add_argument('--netF_param', default='', help="path to netF (to continue training)")
    parser.add_argument('--netDY_param', default='', help="path to netDY (to continue training)")
    parser.add_argument('--netDX_param', default='', help="path to netDX (to continue training)")
    parser.add_argument('--init_gain', type=float, default=0.02,
                        help='scaling factor for normal, xavier and orthogonal.')
    parser.add_argument('--init_type', type=str, default='normal',
                        help='network initialization [normal | xavier | orthogonal]')
    parser.add_argument('--gan_mode', type=str, default='vanilla',
                        help='the type of GAN objective. [vanilla| lsgan | wgan | wgan-gp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
    parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
    parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
    parser.add_argument('--lambda_identity', type=float, default=0.5,
                        help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
    parser.add_argument('--num_iter_D', type=int, default=1, help='number of D iters per each G iter')
    parser.add_argument('--experiment', default='./experiments', help='Where to store samples and models')
    parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
    parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
    parser.add_argument('--ngpu'  , type=int, default=1, help='number of GPUs to use')
    opt = parser.parse_args()
    print(opt)
    return opt
