from dataloader import VPNDataSet, VPNDataSequenceSet
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score,confusion_matrix
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from torch.nn import Module
import torch.nn.functional as F
import torch
from torch import nn
import torch.optim as optim
import numpy as np


def trainStage(model, train_X, train_y, test_X, test_y, param=None):
    model.fit(train_X, train_y)
    acc_train = model.score(train_X, train_y)
    acc_test = model.score(test_X, test_y)
    y_pred = model.predict(test_X)
    recall = recall_score(test_y, y_pred, average='macro')
    print("acc_train: {:.2f}, acc_valid: {:.2f}, recall_valid: {:.2f}".format(acc_train, acc_test, recall))
    return model

class DNN(torch.nn.Module):
    def __init__(self):
        super(DNN, self).__init__()

        self.linear1 = torch.nn.Linear(38, 100)
        self.linear2 = torch.nn.Linear(100, 200)
        self.linear3 = torch.nn.Linear(200, 100)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        return x

def trainRandomForest(output=False):
    # 创建随机森林模型
    model = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=5, max_features=38, n_jobs=32)

    # 定义数据集
    vpnTrainData = VPNDataSet(train=True)
    # vpnTrainData = VPNDataSequenceSet(train=True)
    labels, data = vpnTrainData.getAllData()
    dataNum = len(labels)
    X_train, X_valid_test, y_train, y_valid_test = train_test_split(data, labels, test_size=0.4)
    X_valid, X_test, y_valid, y_test = train_test_split(X_valid_test, y_valid_test, test_size=0.4)
    # scaler = StandardScaler()
    X_train_std = (X_train - np.min(X_train, axis=0)) / (np.max(X_train, axis=0) - np.min(X_train, axis=0) + 1)
    X_valid_std = (X_valid - np.min(X_valid, axis=0)) / (np.max(X_valid, axis=0) - np.min(X_valid, axis=0) + 1)
    X_test_std = (X_test - np.min(X_test, axis=0)) / (np.max(X_test, axis=0) - np.min(X_test, axis=0) + 1)

    # epoch
    epoch = 10
    for i in range(epoch):
        model = trainStage(model, X_train_std, y_train, X_valid_std, y_valid)
    acc_test = model.score(X_test_std, y_test)
    y_pred = model.predict(X_test_std)
    recall = recall_score(y_test, y_pred, average='macro')
    print("acc_test: {}, recall_test: {}".format(acc_test, recall))

    if output:
        vpnTestData = VPNDataSet(train=False)
        test_label, test_data = vpnTestData.getAllData()
        # test_data_std = scaler.fit_transform(test_data)
        test_data_std = (test_data - np.min(test_data, axis=0)) / (np.max(test_data, axis=0) - np.min(test_data, axis=0) + 1)
        y_pred = model.predict(test_data_std)
        f = open('./vpn2-result.txt', 'w')
        for index, (label, y_pre) in enumerate(zip(test_label, y_pred)):
            print("{:06d} {}: {}".format(index, label, y_pre), file=f)
        f.close()

def trainDNN():

    # hypter parameters
    lr = 1e-2
    epoch = 1000
    batch_size = 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # prepare data
    vpndata = VPNDataSet(train=True)
    total_len = len(vpndata)
    train_len = int(total_len * 0.64)
    valid_len = int(total_len * 0.2)
    test_len  = total_len - valid_len - train_len
    trainValidData, testData = torch.utils.data.random_split(vpndata, [train_len + valid_len, test_len])
    trainData, validData = torch.utils.data.random_split(trainValidData, [train_len, valid_len])
    trainDataloader = DataLoader(trainData, batch_size=batch_size, shuffle=True)
    validDataloader = DataLoader(validData, batch_size=batch_size, shuffle=True)
    testDataloader = DataLoader(testData, batch_size=batch_size, shuffle=True)


    model = DNN()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for index in range(epoch):
        y_pred_labels = []
        y_true = []

        for step, (batch_x, batch_y) in enumerate(trainDataloader):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            y_pred = model(batch_x)
            loss = criterion(y_pred, batch_y)

            y_pred = torch.max(y_pred, 1)[1].data
            y_pred = y_pred.cpu()
            y_true.extend([x.item() for x in batch_y])
            y_pred_labels.extend(y_pred.numpy())
            acc = accuracy_score(y_true, y_pred_labels)
            # if (step + 1) % 50 == 0:
            #     print("batch {}, loss: {:.2f}, acc: {:.2%}".format(step +1, loss.item(), acc))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("epoch {}, loss: {:.2f}, acc: {:.2%}".format(index, loss, acc))

        with torch.no_grad():
            y_pred_labels = []
            y_true = []
            for step, (batch_x, batch_y) in enumerate(validDataloader):
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                y_pred = model(batch_x)
                y_pred = torch.max(y_pred, 1)[1].data
                y_pred = y_pred.cpu()
                y_true.extend([x.item() for x in batch_y])
                y_pred_labels.extend(y_pred.numpy())
                acc = accuracy_score(y_true, y_pred_labels)
            print("valid acc: {:.2%}".format(acc))

    with torch.no_grad():
        y_pred_labels = []
        y_true = []
        for step, (batch_x, batch_y) in enumerate(testDataloader):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            y_pred = model(batch_x)
            y_pred = torch.max(y_pred, 1)[1].data
            y_pred = y_pred.cpu()
            y_true.extend([x.item() for x in batch_y])
            y_pred_labels.extend(y_pred.numpy())
            acc = accuracy_score(y_true, y_pred_labels)
        print("test acc: {:.2%}".format(acc))

if __name__ == '__main__':
    trainRandomForest()

