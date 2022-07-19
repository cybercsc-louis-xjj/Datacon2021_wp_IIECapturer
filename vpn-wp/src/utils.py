import numpy as np
import random





def vote():
    with open('./vpn2-result.txt', 'r') as f:
        data = [x.strip().split(' ') for x in f.readlines()]
    result = {}
    for line in data:
        label = line[1][:-1]
        y_pred = int(line[-1])
        if label not in result.keys():
            result[label] = [y_pred]
        else:
            result[label].append(y_pred)
    result_list = []
    for key, value in result.items():
        counts = np.bincount(value)
        label = np.argmax(counts)
        result_list.append((key, label))
    result_list = sorted(result_list, key=lambda x: x[0])
    f = open('result.txt', 'w')
    for item in result_list:
        print("{}.pcap {}".format(item[0], item[1]), file=f)
    f.close()


if __name__ == '__main__':
    randomChange()



