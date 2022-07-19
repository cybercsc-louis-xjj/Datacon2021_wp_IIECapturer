import os
import json
import torch
from torch.utils.data import Dataset
import numpy as np

class VPNDataSet(Dataset):
    """
    加载数据集
    """
    def __init__(self, train=True):
        if train:
            self.path = "/home/sunhanwu/datacon/vpn/stage2data/part2/train_features/"
        else:
            self.path = "/home/sunhanwu/datacon/vpn/stage2data/part2/test_features/"
        self.data = []
        self.train = train
        self.loadData(train)

    def loadData(self, train=True):
        datafiles = [os.path.join(self.path, x) for x in os.listdir(self.path)]
        result = {}
        for file in datafiles:
            if train:
                label = int(file.split('/')[-1].split('.')[0])
            else:
                label = file.split('/')[-1].split('.')[0]
            with open(file, 'r') as f:
                data = json.loads(f.read())
            result[label] = len(data)
            for item in data:
                self.data.append((label, item))
        for key, value in result.items():
            print("{}: {}".format(key, value))

    def getAllData(self):
        labels = [x[0] for x in self.data]
        data = [x[1] for x in self.data]
        if self.train:
            return np.array(labels), np.array(data)
        else:
            return np.array(labels), np.array(data)


    def __getitem__(self, item):
        data = self.data[item][1]
        label = self.data[item][0] - 1
        return torch.tensor(data), torch.tensor(label)

    def __len__(self):
        return len(self.data)

class VPNDataSequenceSet(Dataset):
    """
    序列特征DataSet
    """
    def __init__(self, train=True):
        if train:
            self.path = "/home/sunhanwu/datacon/vpn/stage2data/part2/train_sequence/"
        else:
            self.path = "/home/sunhanwu/datacon/vpn/stage2data/part2/test_sequence/"
        self.train = train
        self.data = []
        self.loadData()

    def loadData(self):
        filenames = [os.path.join(self.path, x) for x in os.listdir(self.path)]
        counter = {}
        for file in filenames:
            with open(file, 'r') as f:
                data = json.loads(f.read())
            if self.train:
                label = int(file.split('/')[-1].split('.')[0])
            else:
                label = 0
            counter[label] = len(data)
            self.data += [(label, x) for x in data]
        for key, value in counter.items():
            print("{}: {}".format(key, value))

    def getAllData(self):
        labels = [x[0] for x in self.data]
        data = [x[1] for x in self.data]
        if self.train:
            return np.array(labels), np.array(data)
        else:
            return None, np.array(data)
        return labels, data

    def __getitem__(self, item):
        data = self.data[item][1]
        label = self.adat[item][0]
        return data, label

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    # vpn = VPNDataSet(train=False)
    vpn = VPNDataSequenceSet(train=True)
