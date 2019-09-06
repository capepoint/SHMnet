import torch
import torch.nn as nn
from torch.utils.data import Dataset,ConcatDataset,DataLoader
import numpy as np
from data import *
class epDataset(Dataset):
    """Sensoring dataset."""
    def __init__(self,label, num_repeats, Normalise=True):
        self.list_IDs = np.arange(num_repeats)
        self.label=label
        self.transform=Normalise
    def __len__(self):
        return len(self.list_IDs)
    def __getitem__(self, idx):
        sdata=np.loadtxt('test_1_acc_4/X_'+str(self.label)+'_1_'+str(idx+1)+'.txt')
        sdata=prep(sdata)
        label=self.label
        return sdata, label
class eptDataset(Dataset):
    """Sensoring dataset."""
    def __init__(self,label, num_repeats, Normalise=True):
        self.list_IDs = np.arange(num_repeats)
        self.label=label
        self.transform=Normalise
    def __len__(self):
        return len(self.list_IDs)
    def __getitem__(self, idx):
        sdata=np.loadtxt('test_1_acc_4/X_'+str(self.label)+'_1_'+str(idx+5) +'.txt')
        sdata=prep(sdata)
        label=self.label
        return sdata, label
class SHMnet(nn.Module):
    def __init__(self,num_classes=10):
        super(SHMnet,self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1,16,kernel_size=7),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(16, 64, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(64, 256, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
	)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 621, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes),
        )
    def forward(self, x):
        x = self.features(x)
        #print ( x.shape)
        x = x.view(x.size(0), 256*621)
        x = self.classifier(x)
        return x
class AlexNet1D(nn.Module):
    def __init__(self, inputsize=5000, num_classes=10):
        super(AlexNet, self).__init__()
        self.inputsize = inputsize
        self.features = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256*155, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0),256*155)
        x = self.classifier(x)
        return x

