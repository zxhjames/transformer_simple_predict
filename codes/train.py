import torch
import torch.nn as nn
import numpy as np
import time
import math
from matplotlib import pyplot
from utils import *

def train(train_data,output_window,epoch,model,scheduler,optimizer,batch_size,criterion,calculate_loss_over_all_values,input_window):
    model.train()  # Turn on the train mode
    total_loss = 0.
    start_time = time.time()

    # 循环选取batch_size
    for batch, i in enumerate(range(0, len(train_data) - 1, batch_size)):
        data, targets = get_batch(train_data, i, batch_size,input_window)
        optimizer.zero_grad()
        output = model(data)
        if calculate_loss_over_all_values:
            loss = criterion(output, targets)
        else:
            loss = criterion(output[-output_window:], targets[-output_window:])
        loss.backward()
        # 梯度裁剪 防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        log_interval = int(len(train_data) / batch_size / 5)
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.6f} | {:5.2f} ms | '
                  'loss {:5.5f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // batch_size, scheduler.get_lr()[0],
                              elapsed * 1000 / log_interval,
                cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


# %%

def plot_and_loss(eval_model, data_source, epoch, scaler,calculate_loss_over_all_values,criterion,output_window,input_window):
    eval_model.eval()
    total_loss = 0.
    test_result = torch.Tensor(0)
    truth = torch.Tensor(0)
    with torch.no_grad():
        for i in range(0, len(data_source) - 1):
            data, target = get_batch(data_source, i, 1,input_window)
            data = data.unsqueeze(1)
            target = target.unsqueeze(1)

            # look like the model returns static values for the output window
            output = eval_model(data)

            if calculate_loss_over_all_values:
                total_loss += criterion(output, target).item()
            else:
                total_loss += criterion(output[-output_window:], target[-output_window:]).item()

            test_result = torch.cat((test_result, output[-1, :].squeeze(1).cpu()),
                                    0)  # todo: check this. -> looks good to me
            truth = torch.cat((truth, target[-1, :].squeeze(1).cpu()), 0)

    # test_result = test_result.cpu().numpy()
    len(test_result)

    print(test_result.size(), truth.size())
    test_result = scaler.inverse_transform(test_result.reshape(-1, 1)).reshape(-1)
    truth = scaler.inverse_transform(truth.reshape(-1, 1)).reshape(-1)

    pyplot.plot(test_result, color="red")
    pyplot.plot(truth[:500], color="blue")
    pyplot.axhline(y=0, color='k')
    pyplot.xlabel("Periods")
    pyplot.ylabel("Y")
    pyplot.savefig('graph/transformer-epoch%d.png' % epoch)
    pyplot.close()
    return total_loss / i


def predict_future(eval_model, data_source, steps, epoch, scaler,input_window,output_window):
    eval_model.eval()
    total_loss = 0.
    test_result = torch.Tensor(0)
    truth = torch.Tensor(0)
    _, data = get_batch(data_source, 0, 1,input_window)
    with torch.no_grad():
        for i in range(0, steps, 1):
            input = torch.clone(data[-input_window:])
            input[-output_window:] = 0
            output = eval_model(data[-input_window:])
            data = torch.cat((data, output[-1:]))

    data = data.cpu().view(-1)

    data = scaler.inverse_transform(data.reshape(-1, 1)).reshape(-1)
    pyplot.plot(data, color="red")
    pyplot.plot(data[:input_window], color="blue")
    pyplot.grid(True, which='both')
    pyplot.axhline(y=0, color='k')
    pyplot.savefig('../assets/transformer-future%d.png' % epoch)
    pyplot.close()


# entweder ist hier ein fehler im loss oder in der train methode, aber die ergebnisse sind unterschiedlich
# auch zu denen der predict_future
def evaluate(eval_model, data_source,output_window,calculate_loss_over_all_values,criterion,input_window):
    eval_model.eval()  # Turn on the evaluation mode
    total_loss = 0.
    eval_batch_size = 1000
    with torch.no_grad():
        for i in range(0, len(data_source) - 1, eval_batch_size):
            data, targets = get_batch(data_source, i, eval_batch_size,input_window)
            output = eval_model(data)
            print(output[-output_window:].size(), targets[-output_window:].size())
            if calculate_loss_over_all_values:
                total_loss += len(data[0]) * criterion(output, targets).cpu().item()
            else:
                total_loss += len(data[0]) * criterion(output[-output_window:], targets[-output_window:]).cpu().item()
    return total_loss / len(data_source)


# %%

def plot(eval_model, data_source, epoch, scaler,calculate_loss_over_all_values,criterion,output_window,input_window):
    eval_model.eval()
    total_loss = 0.
    test_result = torch.Tensor(0)
    truth = torch.Tensor(0)
    with torch.no_grad():
        for i in range(0, len(data_source) - 1):
            data, target = get_batch(data_source, i, 1,input_window)
            data = data.unsqueeze(1)
            target = target.unsqueeze(1)
            # look like the model returns static values for the output window
            output = eval_model(data)
            if calculate_loss_over_all_values:
                total_loss += criterion(output, target).item()
            else:
                total_loss += criterion(output[-output_window:], target[-output_window:]).item()

            test_result = torch.cat((test_result, output[-1, :].squeeze(1).cpu()),
                                    0)  # todo: check this. -> looks good to me
            truth = torch.cat((truth, target[-1, :].squeeze(1).cpu()), 0)

    # test_result = test_result.cpu().numpy()
    len(test_result)

    test_result_ = scaler.inverse_transform(test_result[:700])
    truth_ = scaler.inverse_transform(truth)
    print(test_result.shape, truth.shape)
    for m in range(9):
        test_result = test_result_[:, m]
        truth = truth_[:, m]
        fig = pyplot.figure(1, figsize=(20, 5))
        fig.patch.set_facecolor('xkcd:white')
        pyplot.plot([k + 510 for k in range(190)], test_result[510:], color="red")
        pyplot.title('Prediction uncertainty')
        pyplot.plot(truth[:700], color="black")
        pyplot.legend(["prediction", "true"], loc="upper left")
        ymin, ymax = pyplot.ylim()
        pyplot.vlines(510, ymin, ymax, color="blue", linestyles="dashed", linewidth=2)
        pyplot.ylim(ymin, ymax)
        pyplot.xlabel("Periods")
        pyplot.ylabel("Y")
        #pyplot.show()
        pyplot.savefig('../assets/transformer-future%d.png' % m)
        pyplot.close()
    return total_loss / i

