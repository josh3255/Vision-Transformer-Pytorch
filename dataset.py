import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import os
import numpy as np

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict[b'data'].reshape((-1, 3, 32, 32)).astype(np.float64), dict[b'labels']

class Cifar10Train(Dataset):
    def __init__(self, dataset_path):
        super(Cifar10Train, self).__init__()

        self.data_dict = dict()
        self.dataset_path = dataset_path
        # file_list = os.listdir(self.dataset_path)

        data1, label1 = unpickle(os.path.join(self.dataset_path, 'data_batch_1'))
        data2, label2 = unpickle(os.path.join(self.dataset_path, 'data_batch_2'))
        data3, label3 = unpickle(os.path.join(self.dataset_path, 'data_batch_3'))
        data4, label4 = unpickle(os.path.join(self.dataset_path, 'data_batch_4'))
        data5, label5 = unpickle(os.path.join(self.dataset_path, 'data_batch_5'))

        self.data = np.concatenate([data1, data2, data3, data4, data5], axis=0)
        self.labels = label1 + label2 + label3 + label4 + label5

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class Cifar10Valid(Dataset):
    def __init__(self, dataset_path):
        super(Cifar10Valid, self).__init__()

        self.data_dict = dict()
        self.dataset_path = dataset_path
        # file_list = os.listdir(self.dataset_path)

        self.data, self.label = unpickle(os.path.join(self.dataset_path, 'test_batch'))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]


if __name__ == '__main__':
    dataset = Cifar10Train('/home/josh/Data/cifar-10-python/cifar-10-batches-py/')
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=1)
    print('successfully loaded {} images and labels'.format(len(dataloader.dataset)))

    for step, item in enumerate(dataloader):
        data, label = item
        print(data.shape, label)