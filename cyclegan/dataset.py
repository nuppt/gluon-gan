from __future__ import absolute_import
from __future__ import division
import os
import random
import numpy as np
import mxnet as mx
from mxnet import gluon, nd
from PIL import Image
from transform import *


class UnpairedDataset(gluon.data.Dataset):
    def __init__(self, opt):
        """Initialize this dataset class.

        Initialize:
            - dataset of domain X and Y
            - transformers

        :param opt:  (Option class) -- stores all the experiment flags
        """
        super(UnpairedDataset, self).__init__()
        self.opt = opt
        self.dir_X = os.path.join(opt.data_root, opt.phase + 'A')
        self.dir_Y = os.path.join(opt.data_root, opt.phase + 'B')

        print("dir_X: {}".format(self.dir_X))
        print("dir_Y: {}".format(self.dir_Y))

        self.X_paths = sorted(self._gen_img_path_list(self.dir_X, opt.max_dataset_size))  # load images from '/path/to/data/trainA'
        self.Y_paths = sorted(self._gen_img_path_list(self.dir_Y, opt.max_dataset_size))  # load images from '/path/to/data/trainB'
        self.X_size = len(self.X_paths)  # get the size of dataset A
        self.Y_size = len(self.Y_paths)  # get the size of dataset B

        is_Y_2_X = self.opt.direction == 'Y2X'
        input_nc = self.opt.output_nc if is_Y_2_X else self.opt.input_nc   # get the number of channels of input image
        output_nc = self.opt.input_nc if is_Y_2_X else self.opt.output_nc  # get the number of channels of output image

        self.transformer_X = get_augmentation(self.opt, grayscale=(input_nc == 1))
        self.transformer_Y = get_augmentation(self.opt, grayscale=(output_nc == 1))
        
    def __getitem__(self, index):
        """Return a data point and its meta information

        :param index:
        :return:    a dictionary that contains x, y, x_paths and y_paths
            x   (tensor)        -- an image in the input domain
            y   (tensor)        -- its corresponding image in the target domain
        """
        # 1. choose image by index
        x_path = self.X_paths[index % self.X_size]
        if self.opt.serial_batches:
            index_y = index % self.Y_size
        else:
            index_y = random.randint(0, self.Y_size - 1)
        y_path = self.Y_paths[index_y]

        # 2. load image data
        x_img_nd = nd.array(Image.open(x_path).convert('RGB'))
        y_img_nd = nd.array(Image.open(y_path).convert('RGB'))
        #print("x_img_nd shape: {}".format(x_img_nd.shape))
        #print("y_img_nd shape: {}".format(y_img_nd.shape))


        # 3. apply transformer on image data
        x_img_nd = self.transformer_X(x_img_nd) if self.transformer_X else x_img_nd
        y_img_nd = self.transformer_Y(y_img_nd) if self.transformer_Y else y_img_nd

        #return {'x': x_img_nd, 'y': y_img_nd, 'x_path': x_path, 'y_path': y_path}
        return x_img_nd, y_img_nd

    def __len__(self):
        return max(self.X_size, self.Y_size)

    def _gen_img_path_list(self, dir_name: str, max_size: int) -> list:
        """Generate list of full path in <dir_name>

        :param dir_name:    image directory
        :param max_size:    max limitation of number of image samples
        :return:
        """
        images = []
        assert os.path.isdir(dir_name), '%s is not a valid directory' % dir

        for root, _, fnames in sorted(os.walk(dir_name)):
            for fname in fnames:
                if self._is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)
        return images[:min(max_size, len(images))]

    def _is_image_file(self, filename: str) -> bool:
        """Check type of a file: image or not

        :param filename:
        :return:
        """
        IMG_EXTENSIONS = [
            '.jpg', '.JPG', '.jpeg', '.JPEG',
            '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
        ]
        return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)
