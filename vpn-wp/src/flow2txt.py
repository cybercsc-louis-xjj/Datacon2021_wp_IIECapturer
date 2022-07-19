import os
from joblib import Parallel, delayed

fields = [
    # eth层
    'frame.time',

    # ip层
    'ip.src', 'ip.dst',

    # tcp 层
    'tcp.len',
]


def parsePcap(filename, target):
    """
    使用tshark将pcap处理为txt格式数据
    :param filename: pcap完整路径
    :param target: txt文件存储路径
    :return: None
    """
    global fields
    # format tshark cmd
    cmd = "tshark -r {} -T fields ".format(filename)
    for field in fields:
        cmd += " -e {} ".format(field)

    # format output file path
    if not os.path.exists(target):
        os.makedirs(target)
    targetFile = os.path.join(target, filename.split('/')[-1].split('.')[0])
    cmd += " > {}.txt".format(targetFile)
    print(cmd)
    os.system(cmd)


def multiProcess(dataPath, outputPath, job=30):
    """
    多进程处理
    :param dataPath: 数据目录
    :param outputPath: 处理完之后的数据存放目录
    :return: None
    """
    filenames = [os.path.join(datapath, x) for x in os.listdir(datapath)]
    Parallel(n_jobs=job)(
        (delayed(parsePcap)(file, outputPath) for file in filenames)
    )


if __name__ == '__main__':
    srcPath = "/home/sunhanwu/datacon/vpn/stage2data/part2/test_flows/"
    outputPath = "/home/sunhanwu/datacon/vpn/stage2data/part2/test_txt/"
    dataPaths = [os.path.join(srcPath, x) for x in os.listdir(srcPath)]
    for datapath in dataPaths:
        targetPath = os.path.join(outputPath, datapath.split('/')[-1])
        if not os.path.exists(targetPath):
            os.makedirs(targetPath)
        multiProcess(datapath, targetPath)
