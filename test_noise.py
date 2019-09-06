import torch
import torch.nn as nn
from scipy import signal
import numpy as np
from data import get_data,prep
from torch.autograd import Variable
from torch.utils.data import Dataset,ConcatDataset,DataLoader
from network import *
import matplotlib.pyplot as plt
cuda = True if torch.cuda.is_available() else False
num_class = 11
num_repeats = 5
total=0
correct=0
acc=[]
acc1=[]
model = SHMnet(num_class)

if cuda:
    model=model.cuda()
    model.load_state_dict(torch.load('models/model_params_SHMnet_gpu.pth'))
else:
    model.load_state_dict(torch.load('models/model_params_SHMnet.pth'))
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor
testdataset =[]
for label in range(num_class):
    eptdataset = eptDataset(label,num_repeats+1)
    testdataset = ConcatDataset([testdataset,eptdataset])
viddataloader = DataLoader (testdataset,batch_size =1)
for tests in range(30):
    print('test noise: %d' % epoch)
    total1 = 0
    correct1 = 0
    for i, (sdata, label) in enumerate(viddataloader):
        batch_size = sdata.shape[0]
        sdata = Variable(sdata.type(FloatTensor))
        noise=Variable(torch.from_numpy(np.random.normal(1, 0.5, sdata.shape))).type(FloatTensor)
        x = torch.mul(sdata,noise).view(sdata.size(0),1,sdata.size(1))

        label = Variable (label.type (LongTensor)) 
        _, predicted = torch.max(model(x).data, 1)
        temp = (predicted == label).sum()
        correct += temp
        total += batch_size
        
    print('Accuracy of testing data with 50 percent random Gaussion noise is: %d %%' % (100 * correct / total))
    acc.append(100 * correct / total)
    
np.savetxt('acc-test_noise.txt', acc)


