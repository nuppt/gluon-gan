def transform(data, label):
    return data.astype('float32') / 255., label.astype('float32')
