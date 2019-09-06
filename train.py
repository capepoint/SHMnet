import torch
import torch.nn as nn
from scipy import signal
import numpy as np
from data import get_data, prep
from torch.autograd import Variable
from network import *
import glob
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=1000, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=60, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0001, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.9, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--num_class', type=int, default=11, help='total structral conditions')
parser.add_argument('--num_repeats', type=int, default=4, help='number of repeated tests for training')
parser.add_argument('--path', type=str, default='/home/tz15/workspace/plc_rec_rnn/training_data/train1/train/',help='path of training data')
opt = parser.parse_args()

cuda = True if torch.cuda.is_available() else False
num_class = opt.num_class
num_repeats = opt.num_repeats
total=0
correct=0
acc=[]
acc1=[]
model = SHMnet(num_class)
learning_rate = opt.lr
print(model)
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,betas=(opt.b1,opt.b2))
if cuda:
    model=model.cuda()
    loss=loss.cuda()
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

traindataset = []
for label in range(num_class):
    epdataset = epDataset(label,num_repeats)
    traindataset = ConcatDataset([traindataset,epdataset])
testdataset =[]
for label in range(num_class):
    eptdataset = eptDataset(label,5)
    testdataset = ConcatDataset([testdataset,eptdataset])
dataloader = DataLoader (traindataset, batch_size=opt.batch_size)
viddataloader = DataLoader (testdataset,batch_size = opt.batch_size)
loss_r=[]
start = time.time()

for epoch in range(opt.n_epochs):

    print('Epoch: %d' % epoch)
    total = 0
    correct = 0
    total1 = 0
    correct1 = 0
    for i, (sdata, label) in enumerate(dataloader):
        batch_size = sdata.shape[0]
        sdata = Variable(sdata.type(FloatTensor))
        noise=Variable(torch.from_numpy( np.random.normal(1, 0.2, sdata.shape))).type(FloatTensor)
        x = torch.mul(sdata,noise).view(sdata.size(0),1,sdata.size(1))
        label = Variable (label.type (LongTensor))  
        optimizer.zero_grad()
        output = model(x)
        loss1 = loss(output, label)
        loss1.backward()
        optimizer.step()
        _, predicted = torch.max(model(x).data, 1)
        temp = (predicted == label).sum()
        correct += temp
        total += batch_size
        print(i, loss1.item())
        loss_r.append(loss1.item())
    print('Accuracy of training data is: %d %%' % (100 * correct / total))
    acc.append(100 * correct / total)

    for i, (sdata, label) in enumerate(viddataloader):

        batch_size = sdata.shape[0]
        sdata = Variable(sdata.type(FloatTensor))

        x = sdata.view(sdata.size(0),1,sdata.size(1))
        label = Variable (label.type (LongTensor))  
        _, predicted = torch.max(model(x).data, 1)
        temp = (predicted == label).sum()
        correct1+= temp
        total1+= batch_size
    print ('Accuracy of testing data is: %d %%' % (100 * correct1 / total1))
    acc1.append(100 * correct1 / total1)

end = time.time()
print(end - start)
torch.save(model.state_dict(), 'models/model_params_SHMnet.pth')

