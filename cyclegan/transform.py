from mxnet.gluon.data.vision import transforms
import cv2


class Resize_Aug(object):
    def __init__(self, opt, interpolation=cv2.INTER_CUBIC):
        self.opt = opt
        self.interpolation = interpolation

    def __call__(self, img):
        osize = [self.opt.image_size, self.opt.image_size]
        if 'resize' in self.opt.preprocess:
            return transforms.Resize(osize, interpolation=self.interpolation)(img)
        elif 'scale_width' in self.opt.preprocess:
            return transforms.Resize(osize, keep_ratio=True, interpolation=self.interpolation)(img)
        return img


class Crop_Aug(object):
    def __init__(self, opt, params=None):
        self.opt = opt
        self.params = params

    def __call__(self, img):
        if 'crop' in self.opt.preprocess:
            if self.params is None:
                return transforms.image.random_crop(img, (self.opt.crop_size, self.opt.crop_size))[0]
            else:
                x0, y0, w, h = self.params['crop']
                return transforms.image.fixed_crop(img, x0, y0, w, h, (self.opt.crop_size, self.opt.crop_size))
        return img


class Resize_Power_Aug(object):
    def __init__(self, base, interpolation=cv2.INTER_CUBIC):
        self.base = base
        self.interpolation = interpolation

    def __call__(self, img):
        ow, oh = img.shape[:2]
        h = int(round(oh / self.base) * self.base)
        w = int(round(ow / self.base) * self.base)
        if (h == oh) and (w == ow):
            return img
        return transforms.image.imresize(img, w, h, self.interpolation)


class RandomHorizontalFlip_Aug(object):
    def __init__(self, opt, params=None):
        self.opt = opt
        self.params = params

    def __call__(self, img):
        if not self.opt.no_flip:
            if self.params is None:
                return transforms.RandomFlipTopBottom()(img)
            elif self.params['flip']:
                return transforms.RandomFlipLeftRight()(img)
            else:
                return img
        return img


class Normalization_Aug(object):
    def __init__(self, opt, grayscale):
        self.opt = opt
        self.grayscale = grayscale

    def __call__(self, img):
        if self.grayscale:
            return transforms.Normalize((0.5,), (0.5,))(img)
        return transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img)


def get_augmentation(opt, grayscale):
    return transforms.Compose([
        Resize_Aug(opt),
        Crop_Aug(opt, params=None),
        Resize_Power_Aug(2),
        RandomHorizontalFlip_Aug(opt, params=None),
        transforms.ToTensor(),
        Normalization_Aug(opt, grayscale),
    ])
