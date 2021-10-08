import torch
import torch.nn as nn
import numpy as np
import time
import math
from matplotlib import pyplot
import pandas as pd
from datetime import date
from model import *
from utils import *
from train import *


torch.manual_seed(0)
np.random.seed(0)
'''
teacher forcing旨在防止transformer在解码层出现错误累积的情况
'''
# TODO This concept is also called teacher forceing.
# The flag decides if the loss will be calculted over all
# or just the predicted values.

# 是否计算所有值的loss
calculate_loss_over_all_values = False


debug = False
data_ = pd.read_excel('../data/LD_20142.xlsx', 'Sheet1', parse_dates=["date"],nrows= 1000 if debug else None)

'''
input_window:
output_window:
batch_size:
'''
input_window = 100
output_window = 5
batch_size = 32  # batch size
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
数据预处理
'''
train_data, val_data, scaler = get_data(data_,input_window,output_window,device)
print(train_data.size())
# print(train_data.size(), val_data.size())

# todo这里只取了一个batch数量的数据去做实验
tr, te = get_batch(train_data, 0, batch_size,input_window)

# todo 转换为一个batch的数据大小 tr.shape(100,32,10) te.shape(100,32,10)
print(tr.shape, te.shape)


feature_size=10
num_layers=3
dropout=0.1
model = TransAm(feature_size,num_layers,dropout).to(device)

criterion = nn.MSELoss()
lr = 0.005
# optimizer = torch.optim.SGD(model.parameters(), lr=lr)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
# 学习率调整调度器
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.98)
best_val_loss = float("inf")
epochs = 20  # The number of epochs
best_model = None

# 进行20轮迭代
for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train(train_data,output_window,epoch,model,scheduler,optimizer,batch_size,criterion,calculate_loss_over_all_values,input_window)
    if (epoch % 1 is 0):
        val_loss = plot(model, val_data,  epoch, scaler,calculate_loss_over_all_values,criterion,output_window,input_window)
        # eval_model, data_source, steps, epoch, scaler,input_window,output_window
        # predict_future(model, val_data,200,epoch,scaler,input_window,output_window)
    else:
        val_loss = evaluate(model, val_data,output_window,calculate_loss_over_all_values,criterion,input_window)

    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.5f} | valid ppl {:8.2f}'.format(epoch, (
                time.time() - epoch_start_time),
                                                                                                  val_loss,
                                                                                                  math.exp(val_loss)))
    print('-' * 89)

    # if val_loss < best_val_loss:
    #    best_val_loss = val_loss
    #    best_model = model

    scheduler.step()