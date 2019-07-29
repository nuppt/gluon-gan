from mxnet.gluon.data.vision import transforms


def transform(data, label):
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0, 1)
    ])
    data = data_transform(data)
    label = label.astype('float32')

    return data, label