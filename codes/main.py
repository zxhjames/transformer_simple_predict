'''
Author: your name
Date: 2021-10-04 15:32:19
LastEditTime: 2021-10-06 08:46:49
LastEditors: your name
Description: In User Settings Edit
FilePath: \project_demo\研二\实验basemodel\transformer_ETL\codes\main.py
'''
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

# TODO This concept is also called teacher forceing.
# The flag decides if the loss will be calculted over all
# or just the predicted values.
calculate_loss_over_all_values = False

debug = True
data_ = pd.read_excel('../data/LD_20142.xlsx', 'Sheet1', parse_dates=["date"],nrows= 10000 if debug else None)

input_window = 100
output_window = 5
batch_size = 32  # batch size
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_data, val_data, scaler = get_data(data_,input_window,output_window,device)
print(train_data.size())
# print(train_data.size(), val_data.size())
tr, te = get_batch(train_data, 0, batch_size,input_window)
print(tr.shape, te.shape)

model = TransAm().to(device)

criterion = nn.MSELoss()
lr = 0.005
# optimizer = torch.optim.SGD(model.parameters(), lr=lr)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.98)

best_val_loss = float("inf")
epochs = 20  # The number of epochs
best_model = None

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train(train_data,output_window,epoch,model,scheduler,optimizer,batch_size,criterion,calculate_loss_over_all_values,input_window)

    if (epoch % 1 is 0):
        val_loss = plot(model, val_data,  epoch, scaler,calculate_loss_over_all_values,criterion,output_window,input_window)
        # predict_future(model, val_data,200,epoch,scaler)
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