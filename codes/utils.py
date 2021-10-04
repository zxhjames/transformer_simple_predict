from datetime import date
from sklearn.preprocessing import MinMaxScaler
import torch
import numpy as np
import pandas as pd
# if window is 100 and prediction step is 1
# in -> [0..99]
# target -> [1..100]
def create_inout_sequences(input_data, tw,output_window):
    inout_seq = []
    L = len(input_data)
    # print(L)
    np.zeros((output_window, 3))
    for i in range(L - tw):
        train_seq = np.append(input_data[i:i + tw, :][:-output_window, :], np.zeros((output_window, 10)), axis=0)
        train_label = input_data[i:i + tw, :]
        # print(train_seq.shape,train_label.shape)
        # train_label = input_data[i+output_window:i+tw+output_window]
        inout_seq.append((train_seq, train_label))
    return torch.FloatTensor(inout_seq)


def get_data(data_,input_window,output_window,device):
    time = np.arange(0, 400, 0.1)
    # amplitude   = np.sin(time) + np.sin(time*0.05) +np.sin(time*0.12) *np.random.normal(-0.2, 0.2, len(time))
    # series = read_csv('daily-min-temperatures.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    data = data_.loc[
        (data_["date"] >= pd.Timestamp(date(2014, 1, 1))) & (data_["date"] <= pd.Timestamp(date(2014, 2, 10)))]
    data = data.loc[:, "MT_200":  "MT_209"]
    series = data.to_numpy()
    # print('a',series.shape)
    amplitude = scaler.fit_transform(series)
    # print('b', amplitude.shape)
    # amplitude = scaler.fit_transform(amplitude.reshape(-1, 1)).reshape(-1)

    # print(amplitude.shape)
    sampels = 2800
    train_data = amplitude[:sampels]
    test_data = amplitude[sampels:]

    # print(train_data.shape,test_data.shape)
    # convert our train data into a pytorch train tensor
    # train_tensor = torch.FloatTensor(train_data).view(-1)
    # todo: add comment..
    # print('c',train_data.shape)

    train_sequence = create_inout_sequences(train_data, input_window,output_window)
    # print('a',train_sequence.size())
    train_sequence = train_sequence[:-output_window]  # todo: fix hack?

    # test_data = torch.FloatTensor(test_data).view(-1)
    test_data = create_inout_sequences(test_data, input_window,output_window)
    test_data = test_data[:-output_window]  # todo: fix hack?

    return train_sequence.to(device), test_data.to(device), scaler


def get_batch(source, i, batch_size,input_window):
    seq_len = min(batch_size, len(source) - 1 - i)
    data = source[i:i + seq_len]
    input = torch.stack(torch.stack([item[0] for item in data]).chunk(input_window, 1)).squeeze()  # 1 is feature size
    target = torch.stack(torch.stack([item[1] for item in data]).chunk(input_window, 1)).squeeze()
    return input, target
