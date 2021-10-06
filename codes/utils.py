from datetime import date
from sklearn.preprocessing import MinMaxScaler
import torch
import numpy as np
import pandas as pd
# if window is 100 and prediction step is 1
# in -> [0..99]
# target -> [1..100]

# todo 创建输入序列,
def create_inout_sequences(input_data, tw,output_window):
    inout_seq = []
    L = len(input_data)
    # print(L)
    np.zeros((output_window, 3))
    # todo input_data是一个shape为(2800, 10) 的ndarray,将时间步往前转移一步，然后依次迭代
    for i in range(L - tw):
        # todo 注意这里的train_seq是(95,10)的shape  output_window是(5,10)的shape 将这两个矩阵进一维的拼接，这里留出的五个位置都用0初始化，
        train_seq = np.append(input_data[i:i + tw, :][:-output_window, :], np.zeros((output_window, 10)), axis=0)
        # todo 这里的train_label代表的是真实的值，最后的五个位置用真实的值替代
        train_label = input_data[i:i + tw, :]
        # print(train_seq.shape,train_label.shape)
        # train_label = input_data[i+output_window:i+tw+output_window]
        inout_seq.append((train_seq, train_label))
    return torch.FloatTensor(inout_seq)

'''
数据预处理:
'''
def get_data(data_,input_window,output_window,device):
    # time = np.arange(0, 400, 0.1)
    # amplitude   = np.sin(time) + np.sin(time*0.05) +np.sin(time*0.12) *np.random.normal(-0.2, 0.2, len(time))
    # series = read_csv('daily-min-temperatures.csv', header=0, index_col=0, parse_dates=True, squeeze=True)

    # todo 归一化
    scaler = MinMaxScaler(feature_range=(-1, 1))
    # 取其中一部分数据
    data = data_.loc[
        (data_["date"] >= pd.Timestamp(date(2014, 1, 1))) & (data_["date"] <= pd.Timestamp(date(2014, 2, 10)))]
    # 取其中一部分列
    data = data.loc[:, "MT_200":  "MT_209"]

    # 获取归一化之后的数组
    amplitude = scaler.fit_transform(data.to_numpy())
    # print('b', amplitude.shape)
    # amplitude = scaler.fit_transform(amplitude.reshape(-1, 1)).reshape(-1)
    # print(amplitude.shape)

    # 划分训练集和测试集
    sampels = 2800
    train_data = amplitude[:sampels]
    test_data = amplitude[sampels:]
    # print(train_data.shape,test_data.shape)
    # convert our train data into a pytorch train tensor
    # train_tensor = torch.FloatTensor(train_data).view(-1)
    # todo: add comment..
    # print('c',train_data.shape)

    # todo 创建训练集序列
    train_sequence = create_inout_sequences(train_data, input_window,output_window)
    # print('a',train_sequence.size())

    # todo 取前output_size长度的训练集序列
    train_sequence = train_sequence[:-output_window]  # todo: fix hack?
    # test_data = torch.FloatTensor(test_data).view(-1)
    test_data = create_inout_sequences(test_data, input_window,output_window)
    test_data = test_data[:-output_window]  # todo: fix hack?
    return train_sequence.to(device), test_data.to(device), scaler


# 将数据转换为每一个batch下的数据
# todo params [train_data, 0, batch_size,input_window]
def get_batch(source, i, batch_size,input_window):
    seq_len = min(batch_size, len(source) - 1 - i)
    # data (32,2,100,10)
    data = source[i:i + seq_len]
    # torch stack函数 chunk函数
    '''
    stack:例如 a = (3,32,32) b = (3,32,32) 使用stack在哪一个维度上进行拼接，那就在当前维度增加一个拼接维度的倍数，例如torch.stack([a,b],axis=0).shape = (2,3,32,32)
    chunk:例如c=torch.tensor([[1,4,7,9,11],[2,5,8,9,13]])，print(torch.chunk(c,3,1))，输出结果为：(tensor([[1, 4],[2, 5]]), tensor([[7, 9],[8, 9]]), tensor([[11], [13]]))
    squeeze:将shape中维度为1的去掉
    '''
    input = torch.stack(torch.stack([item[0] for item in data]).chunk(input_window, 1)).squeeze()  # 1 is feature size
    target = torch.stack(torch.stack([item[1] for item in data]).chunk(input_window, 1)).squeeze()
    return input, target
