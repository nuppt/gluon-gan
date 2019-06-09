from mxnet.gluon.data.vision import transforms
from PIL import Image


def get_transform(opt, params=None, grayscale=False, method=Image.BICUBIC, convert=True):
    return None
    # transform_list = []
    #
    # if grayscale:
    #     transform_list.append(transforms.Grayscale(1))
    #
    # if 'resize' in opt.preprocess:
    #     osize = [opt.load_size, opt.load_size]
    #     transform_list.append(transforms.Resize(osize, method))
    # elif 'scale_width' in opt.preprocess:
    #     transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.load_size, method)))
    #
    # if 'crop' in opt.preprocess:
    #     if params is None:
    #         transform_list.append(transforms.RandomCrop(opt.crop_size))
    #     else:
    #         transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.crop_size)))
    #
    # if opt.preprocess == 'none':
    #     transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base=4, method=method)))
    #
    # if not opt.no_flip:
    #     if params is None:
    #         transform_list.append(transforms.RandomHorizontalFlip())
    #     elif params['flip']:
    #         transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))
    #
    # if convert:
    #     transform_list += [transforms.ToTensor()]
    #     if grayscale:
    #         transform_list += [transforms.Normalize((0.5,), (0.5,))]
    #     else:
    #         transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    # return transforms.Compose(transform_list)