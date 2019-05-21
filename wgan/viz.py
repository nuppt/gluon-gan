import math
import numpy as np


def save_images(images, filename):
    from PIL import Image
    row = int(math.sqrt(len(images)))
    col = row
    height = sum(image.shape[0] for image in images[0:row])
    width = sum(image.shape[1] for image in images[0:col])
    output = np.zeros((height, width, 3))

    for i in range(row):
        for j in range(col):
            image = images[i*row+j]
            h, w, d = image.shape
            output[i*h:i*h + h, j*w:j*w+w] = image
    output = (output * 255).clip(0,255).astype('uint8')
    im = Image.fromarray(output)
    im.save(filename)
