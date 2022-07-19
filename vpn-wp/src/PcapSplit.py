from pcap_splitter.splitter import PcapSplitter
import os
from joblib import Parallel, delayed


def splitPcap(filename):
    """
    将pcap 包按照session拆分开
    :param filename: pcap文件完整路径
    :param output: 输出位置目录
    :return: None
    """
    global output
    ps = PcapSplitter(filename)
    pcapname = filename.split('/')[-1].split('.')[0]
    target = os.path.join(output, pcapname)
    print("[ Pcap Split ] {}.pcap --> {}".format(pcapname, target))
    if not os.path.exists(target):
        os.makedirs(target)
    ps.split_by_session(target)


def mutilProcess(filePath, job=32):
    """
    使用joblib多进程处理
    :param job: 进程数量, 默认32
    :return: None
    """
    filenames = [os.path.join(filePath, x) for x in os.listdir(filePath)]
    Parallel(n_jobs=job)(
        (delayed(splitPcap)(filename) for filename in filenames)
    )


if __name__ == '__main__':
    output = "/home/sunhanwu/datacon/vpn/stage2data/part2/test_flows/"
    datapath = "/home/sunhanwu/datacon/vpn/stage2data/part2/test_data/"
    mutilProcess(datapath, job=30)
