import torch
import numpy as np

from array_tool import to_categorical
from model import Unet
from dataloader import label_load, flow_load
import random


label_collapsed = label_load()
labels = []
for label in label_collapsed:
    labels.append(label[2])
flow_root = '../../data/Motion_Emotion_Dataset/flows'
flows, flow_index = flow_load(flow_root)

net = Unet()
net = net.to(net.device)
y = torch.from_numpy(to_categorical([l-1 for l in labels], 5)).to(net.device)

epoch_size = 500
batch_size = 400

# manually split train & test videos
# take the first 25 videos as the training dataset, the last 6 videos as the test dataset
train_vid_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
test_vid_idx = [25, 26, 27, 28, 29, 30]

idx = 0
while label_collapsed[idx, 0] <= train_vid_idx[-1]:
    idx += 1

train_idx = [i for i in range(idx)]
test_idx = [i+idx for i in range(label_collapsed.shape[0]-idx)]


batches = len(train_idx)//batch_size
for ep in range(epoch_size):

    net.train()

    ep_idx = [i for i in range(len(train_idx))]
    random.shuffle(ep_idx)

    for b in range(batches):
        batch_x= flows[ep_idx[b*batch_size:b*batch_size+batch_size]].astype(np.float32)
        batch_x = batch_x.transpose(0, 3, 1, 2)
        batch_x = torch.from_numpy(batch_x).float()
        batch_x = batch_x.to(net.device)

        batch_y = y[ep_idx[b*batch_size:b*batch_size+batch_size]]

        o = net(batch_x)
        pred = torch.argmax(o, 1)
        truth = torch.argmax(batch_y, 1)
        acc = pred.eq(truth).sum().item() / (truth.shape[0] + 0.001)
        loss = net.loss(o, batch_y)
        train_loss = loss.item()

        # bp
        net.optim.zero_grad()
        loss.backward()
        net.optim.step()

        print('ep {} batch {}, loss = {}, acc = {}'.format(ep, b, loss.item(), acc))


    net.eval()

    val_batch_size = 200
    val_batches = len(test_idx) //val_batch_size
    acc_vals = []
    loss_vals = []
    for b in range(val_batches):
        batch_x= flows[test_idx[b*val_batch_size:b*val_batch_size+val_batch_size]].astype(np.float32)
        batch_x = batch_x.transpose(0, 3, 1, 2)
        batch_x = torch.from_numpy(batch_x).float()
        batch_x = batch_x.to(net.device)

        batch_y = y[test_idx[b*val_batch_size:b*val_batch_size+val_batch_size]]

        o = net(batch_x)
        pred = torch.argmax(o, 1)
        truth = torch.argmax(batch_y, 1)
        acc = pred.eq(truth).sum().item() / (truth.shape[0] + 0.001)
        loss = net.loss(o, batch_y)
        loss_val = loss.item()

        acc_vals.append(acc)
        loss_vals.append(loss_val)

    acc_avg_val = np.average(np.array(acc_vals))
    loss_avg_val = np.average(np.array(loss_vals))
    print('ep {} val loss = {}, acc = {}'.format(ep, loss_avg_val, acc_avg_val))

