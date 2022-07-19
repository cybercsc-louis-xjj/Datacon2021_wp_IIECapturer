import torch
import time
import datetime
import json
import random import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision as tv
import torchvision.transforms as transforms
from torch.utils.data import Dataset,DataLoader
from sklearn.metrics import accuracy_score,confusion_matrix
import sys
import os



############################################################
#                DEFINE Model
############################################################

class LSTM(nn.Module):
    # 构造函数，接受一个参数字典已进行初始化操作
    def __init__(self,para):
        super(LSTM,self).__init__()
        '''
        Embedding层：
        将每个特征嵌入到embedding_dim_payload维的空间中，形成embedding_dim_payload维的向量
        必须参数1：要embedding的特征数目
        必须参数2：每个特征要扩展的维数
        '''
        # todo: 需要仔细分析
        # 应该不需要进行embedding
        self.embedding_payload = nn.Embedding(256, 256) # 256
        self.embedding_interval = nn.Embedding(para['vocab_size_win'], para['embedding_dim_win']) # 10
        self.embedding_packet = nn.Embedding(para['vocab_size_packet'], para['embedding_dim_packet']) # 10
        # self.normal_inter = nn.LayerNorm((batch_size, para['vocab_size_inter']))
        '''
        lstm层：
        input_size是输入到每一个time_step中特征的维度，数值上等于上一层特征在embedding后的维数
        hidden是每一个time_step中济进行运算的神经元的个数
        num_layers是指run的结构堆叠的层数
        lstm_output.shape: (batch_size, equence_len, num_directions * hidden_size)
            num_directions: 指单双向lstm网络，单向的话为1，双向的话为2
        h_n.shape: (num_layers * num_directions, batch_size, hidden_size)
            num_layer: 指lstm网络堆叠的层数
        h_c.shape: (num_layers * num_directions, batch_size, hidden_size)
        '''
        self.lstm_payload = nn.LSTM(
            input_size  = para['embedding_dim_payload'], # 256
            hidden_size = para['hidden_size_payload'], # 200
            num_layers  = para['num_layers'], # 3层
            batch_first = True
        )
        self.lstm_win = nn.LSTM(
            input_size = para['embedding_dim_win'],
            hidden_size = para['hidden_size_win'],
            num_layers = para['num_layers'],
            batch_first = True
        )
        self.lstm_packet = nn.LSTM(
            input_size = para['embedding_dim_packet'],
            hidden_size = para['hidden_size_packet'],
            num_layers = para['num_layers'],
            batch_first = True
        )
        #  self.lstm_inter = nn.LSTM(
            #  input_size = 1,
            #  hidden_size= para['hidden_size_inter'],
            #  num_layers = para['num_layers'],
            #  batch_first = True
        #  )
        self.lstm_final = nn.LSTM(
            input_size = para['embedding_dim_final'],
            hidden_size= para['hidden_size_final'],
            num_layers=para['num_layers'],
            batch_first=True
        )
        '''
        normal层:
        '''
        #  self.normal = torch.nn.LayerNorm((batch_size,para['hidden_size_final']))
        #  self.normal = torch.nn.LayerNorm((batch_size,para['hidden_size_payload'] + para['hidden_size_win']))
        self.normal = torch.nn.LayerNorm((batch_size,para['hidden_size_packet']))
        '''
        线性层：
        将lstm层最后一个time_step的结果输入到一个线性层（Linear）和一个Relu激活函数组成的输出层
        Linear:
            第一个参数是上一层的神经元的个数，也就是lstm的隐藏层的神经元个数
            第二个参数是要输出的个数，我们要坐的事TopNum分类，所以数值上等于TopNum
        '''
        #  self.linear = nn.Sequential(nn.Linear(para['hidden_size_final'],para['label_size']))
        #  self.linear = nn.Sequential(nn.Linear(para['hidden_size_payload'] + para['hidden_size_win'],para['label_size']))
        self.linear = nn.Sequential(nn.Linear(para['hidden_size_packet'],para['label_size']))

    def forward(self, x):
        '''
        # x.shape: (batch_size, sequence_len)
        '''
        #  ipdb.set_trace()
        x_payload = x[:,:PAYLOAD_LEN].type(torch.cuda.LongTensor)
        #  x_win = x[:,PAYLOAD_LEN:PAYLOAD_LEN+WIN_LEN].type(torch.cuda.LongTensor)
        #  x_packet = x[:,:PACKET_LEN].type(torch.cuda.LongTensor)
        #  x_payload = x[:,:PAYLOAD_LEN].type(torch.cuda.LongTensor)
        #  x_win = x[:,PAYLOAD_LEN:].type(torch.cuda.LongTensor)
        #  x_packet = x[:,PAYLOAD_LEN+WIN_LEN:].type(torch.cuda.LongTensor)
        #  x_packet = x[:,PAYLOAD_LEN + WIN_LEN:PAYLOAD_LEN + WIN_LEN + INTER_LEN].type(torch.cuda.LongTensor)
        #  x_inter = x[:,PAYLOAD_LEN + WIN_LEN + INTER_LEN:]
        #  x_win = x[:,:WIN_LEN].type(torch.cuda.LongTensor)
        #  x_packet = x[:,WIN_LEN:].type(torch.cuda.LongTensor)
        '''
        # x_payload.shape: (batch_size,payload_len)
        # x_win.shape: (batch_size,win_len)
        # x_packet.shape: (batch_size,packet_len)
        '''
        embedding_payload = self.embedding_payload(x_payload)
        #  embedding_win = self.embedding_win(x_win)
        #  embedding_packet = self.embedding_packet(x_packet)
        #  normal_inter = self.normal_inter(x_inter)
        #  embedding_inter = normal_inter.view(batch_size,-1,1)
        '''
        # embedding_payload.shape: (batch_size,payload_len, embedding_dim_payload)
        # embedding_win.shape: (batch_size,sequence_len, embedding_dim_win)
        # embedding_packet.shape: (batch_size,sequence_len, embedding_dim_packet)
        '''
        payload_lstm_output,(h_n,h_c)= self.lstm_payload(embedding_payload)
        #  win_lstm_output,(h_n,h_c)= self.lstm_win(embedding_win)
        #  packet_lstm_output,(h_n,h_c)= self.lstm_packet(embedding_packet)
        #  inter_lstm_output,(h_n,h_c) = self.lstm_inter(embedding_inter)
        '''
        # payload_lstm_output.shape: (batch_size, payload_len, num_directions * hidden_size_payload)
        # win_lstm_output.shape: (batch_size, win_len, num_directions * hidden_size_win)
        # packet_lstm_output.shape: (batch_size, packet_len, num_directions * hidden_size_packet)
        '''
        #  final_output_cat = torch.cat((payload_lstm_output,win_lstm_output,packet_lstm_output),dim=1)
        #  final_output_cat = torch.cat((win_lstm_output,packet_lstm_output),dim=1)
        #  final_lstm_output,(h_n,h_c) = self.lstm_final(final_output_cat) #最后加一层lstm
        # last_time_step_output_payload = payload_lstm_output[:,-1,:]
        # last_time_step_output_win = win_lstm_output[:,-1,:]
        # last_time_step_output_packet = packet_lstm_output[:,-1,:]
        '''
        # last_time_step_output_payload.shape: (batch_size,num_directions * hidden_size_payload)
        # last_time_step_output_win.shape: (batch_size,num_directions * hidden_size_win)
        # last_time_step_output_packet.shape: (batch_size,num_directions * hidden_size_packet)
        '''
        last_time_step_output_payload = payload_lstm_output[:,-1,:]
        #  last_time_step_output_final = final_lstm_output[:,-1,:]
        #  last_time_step_output_win = win_lstm_output[:,-1,:]
        #  last_time_step_output_packet = packet_lstm_output[:,-1,:]
        #  cat_last_time_step = torch.cat((last_time_step_output_final,last_time_step_output_win,last_time_step_output_packet),dim = 1)
        # last_time_step_output = torch.cat((last_time_step_output_payload,last_time_step_output_win,last_time_step_output_packet),1)
        # last_time_step_output.shape: (batch_size,num_directions * (hidden_szie_payload + hidden_size_win + hidden_size_packet))
        #  normal_output = self.normal(cat_last_time_step)
        normal_output = self.normal(last_time_step_output_payload)
        # normal_output.shape: (batch_size,num_directions * (hidden_szie_payload + hidden_size_win + hidden_size_packet))
        linear_output = self.linear(normal_output)
        # linear_output.shape: (batch_size, TopNum)
        return linear_output

############################################################
#                RUN Model
############################################################

if __name__ == '__main__':


    # use GPU if it is available,otherwise use cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.enabled = False
    #  device = torch.device("cpu")
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    '''
    准备数据，数据共分为三个部分
    训练集: 占总数据量的64%
    验证集: 占总数据量的16%
    测试集: 占总数据量的20%
    '''
    filenames = [x for x in os.listdir(DATA_PATH['resample']) if 'json' in x]
    # filenames = ['activities-0-fields-tcpflow-format.json'] # for debug
    iotdata = IoTDataSet(filenames)
    # 计算各个部分数据的量
    iotdata_len = len(iotdata) #数据总量
    train_len = int(iotdata_len * 0.64)
    test_len = int(iotdata_len * 0.2) # 测试集
    valid_len = iotdata_len - train_len - test_len # 验证集
    train_valid_len = train_len + valid_len # 训练验证集数量
    print('total data:',len(iotdata))
    print('train:{},valid:{},test:{}'.format(train_len,valid_len,test_len))

    # 拆分训练接、验证集和测试集
    train_valid_data,test_data = torch.utils.data.random_split(iotdata,[train_valid_len,test_len])
    train_data,valid_data = torch.utils.data.random_split(train_valid_data,[train_len,valid_len])
    train_loader = DataLoader(train_data,batch_size=batch_size,shuffle= True,drop_last=True)
    valid_loader = DataLoader(valid_data,batch_size=batch_size,shuffle=True,drop_last=True)
    test_loader = DataLoader(test_data,batch_size=batch_size,shuffle=True,drop_last= True)

    # 实例化模型
    lstm  = LSTM(NET_CONFIG)
    #  lstm = nn.DataParallel(lstm) #并行运算,并行计算的时候出现一堆警告信息，暂时没有解决
    lstm.to(device)

    # train
    print('device:',device)
    print('[training...]')
    criterrion = torch.nn.CrossEntropyLoss(size_average = False) # 损失函数使用交叉熵函数
    optimizer = torch.optim.Adam(lstm.parameters(),lr=learning_rate,betas=(0.9,0.99)) # 优化器使用adam优化器,weight_decay是L2正则项的
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,gamma=0.1,step_size= 2)
    startTime = time.time()
    for epoch in range(epoch_size):
        y_pred_labels = []
        y_true = []
        # 开启训练模式
        lstm.train()
        for step,(batch_x,batch_y) in enumerate(train_loader):
            # batch_x.shape: (batch_size, sequence_len)
            # batch_y.shape: (batch_size)
            batch_x = batch_x.to(device)
            y_pred = lstm(batch_x)
            # y_pred.shape: (batch_size, label_size)
            batch_y = batch_y.to(device)
            '''
            这里使用交叉熵损失函数
            交叉熵损失函数可以计算linear输出的多个值与目标值之间的差距，就像二分类的那个
            '''
            ipdb.set_trace()
            loss = criterrion(y_pred,batch_y)
            '''
            torch.max(y_pred,1)会在y_pred的第1维方向上计算最大值
            返回两个数据，一个是最大值的数值，一个是最大值的位置索引
            所以这里的最大值的位置就代表了分类的结果
            '''
            y_pred = torch.max(y_pred,1)[1].data
            y_pred = y_pred.cpu()

            y_true.extend([x.item() for x in batch_y])
            # list.extend 表示从可迭代对象中添加元素
            y_pred_labels.extend(y_pred.numpy())
            acc = accuracy_score(y_true,y_pred_labels)
            if (step+1) % LogIter == 0:
                # print('\t[ batch {} / {} ] loss:{:.2f},acc:{:.2%},learning_rate:{}'.format(step,len(train_loader),loss.item(),acc,optimizer.state_dict()['param_groups'][0]['lr']))
                print('\t[ batch {} / {} ] loss:{:.2f},acc:{:.2%}'.format(step + 1,len(train_loader),loss.item(),acc))
                # print(loss.grad)
            optimizer.zero_grad()
            loss.backward()
            # # 进行梯度剪裁，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(lstm.parameters(),max_norm=MAX_NORM,norm_type=2)
            optimizer.step()
        print('[ epoch {} / {}] train loss:{:.2f},acc:{:.2%}\t'.format(epoch,epoch_size,loss.item(),acc),end='')

        # valid
        lstm.eval()
        with torch.no_grad():
            y_pred_labels = []
            y_true = []
            for step,(batch_x,batch_y) in enumerate(valid_loader):
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                y_pred = lstm(batch_x)
                # loss = criterrion(y_pred,batch_y)
                y_pred = torch.max(y_pred,1)[1].data
                y_pred = y_pred.cpu()
                y_true.extend([x.item() for x in batch_y])
                y_pred_labels.extend(y_pred.numpy())
                # ipdb.set_trace()
                acc = accuracy_score(y_true,y_pred_labels)
            print('vaild acc:{:.2%}'.format(acc))

    # test
    endTime = time.time()
    print("time:",endTime-startTime)
    print('[testing...]')
    y_true = []
    y_pred_labels = []
    start_time = time.time()
    with torch.no_grad():
        for batch_x,batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            y_pred = lstm(batch_x)
            y_pred = torch.max(y_pred,1)[1].data
            y_pred = y_pred.cpu()
            y_pred_labels.extend(y_pred.numpy())
            y_true.extend([x.item() for x in batch_y])
    end_time = time.time()
    print('test acc:{:.2%}'.format(accuracy_score(y_true,y_pred_labels)))
    print('test :{},time elapse:{}'.format(end_time - start_time,test_len))
    confusion = confusion_matrix(y_true,y_pred_labels)
    # plotCM([i for i in range(TopNum)],confusion,'/home/sunhanwu/test.jpg')
    labels = iotdata.top

