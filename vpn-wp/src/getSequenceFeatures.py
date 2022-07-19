import os
from joblib import Parallel, delayed
import json
import numpy as np

def getSingleLabelSequenceFeatures(lableDir):
    """
    解析flow特征文件
    :param lableDir: 数据文件夹
    :return: list
    """
    # tcp payload data
    tcpPayloadFile = os.path.join(lableDir, "tcpPayLoad.txt")
    with open(tcpPayloadFile, 'r') as f:
        tcpPayloadData = [x.strip().split(' ') for x in f.readlines()]

    packetLengthFile = os.path.join(lableDir, "packetLength.txt")
    with open(packetLengthFile, 'r') as f:
        packetLengthData = [x.strip().split(' ') for x in f.readlines()]

    intervalFile = os.path.join(lableDir, "packetTime.txt")
    with open(intervalFile, 'r') as f:
        intervalData = [x.strip().split(' ') for x in f.readlines()]

    result = []
    for i in range(len(tcpPayloadData)):
        temp = []
        temp += tcpPayloadData[i][:32]
        temp += packetLengthData[i]
        temp += intervalData[i]
        result.append(temp)
    return result


def getAllLabelSequenceFeatures(label, train=True):
    """
    获取某个label的全部序列特征，并存储在一个json文件中
    :param label: lable
    :param train:
    :return: None
    """
    global trainSequenceFeaturesPath
    global testSequenceFeaturesPath
    if train:
        dataPath = trainSequenceFeaturesPath
    else:
        dataPath = testSequenceFeaturesPath
    labelDirs = [os.path.join(dataPath, x) for x in os.listdir(dataPath) if x.startswith("{}".format(label))]
    result = []
    for labelDir in labelDirs:
        temp = getSingleLabelSequenceFeatures(labelDir)
        result += temp
    if train:
        targetDir = "/home/sunhanwu/datacon/vpn/stage2data/part2/train_sequence/"
    else:
        targetDir = "/home/sunhanwu/datacon/vpn/stage2data/part2/test_sequence/"
    result_str = json.dumps(result)
    targetFile = os.path.join(targetDir, "{}.json".format(label))
    print("{} done".format(targetFile))
    with open(targetFile, 'w') as f:
        f.write(result_str)

def multiProcess(job=32, train=True):
    """

    :param job:
    :param train:
    :return:
    """
    if train:
        Parallel(n_jobs=job)(
            (delayed(getAllLabelSequenceFeatures)(label, True) for label in range(1, 101))
        )
    else:
        Parallel(n_jobs=job)(
            (delayed(getAllLabelSequenceFeatures)(label, False) for label in os.listdir("/home/lixiang/datacon/feature_txt/test_flows/"))
        )




if __name__ == '__main__':
    trainSequenceFeaturesPath = "/home/lixiang/datacon/feature_txt/train_flows/"
    testSequenceFeaturesPath = "/home/lixiang/datacon/feature_txt/test_flows/"
    multiProcess(32, False)
