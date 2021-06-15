# coding:utf8
from torch import nn
from .basic_module import BasicModule
import torch


class CNN(BasicModule):
    def __init__(self,feature_size,hidden,out_feature,window_sizes,input_size,sequence_length):
        super(CNN, self).__init__()
        self.model_name = 'CNN'
        self.convs = nn.ModuleList([
                nn.Sequential(nn.Conv1d(in_channels = input_size, 
                                        out_channels = feature_size, 
                                        kernel_size=h),
                              nn.ReLU(),
                              nn.MaxPool1d(kernel_size=sequence_length-h+1))
                     for h in window_sizes
                ])
        self.sequence_length = sequence_length
        self.input_size = input_size
        
        self.regression = nn.Sequential()
        self.regression.add_module('linear0',nn.Linear(feature_size*len(window_sizes), hidden[0]))
        self.regression.add_module('relu0', nn.ReLU())
        for i , hid in enumerate(hidden):
            if i == len(hidden) - 1:
                self.regression.add_module(('linear'+str(i+1)) , nn.Linear(hidden[i], out_feature ))
            else:
                self.regression.add_module(('linear'+str(i+1)), nn.Linear(hidden[i], hidden[i+1]))
                self.regression.add_module(('relu' + str(i+1)), nn.ReLU())
        
        
    def forward(self,x,device):
        #x = x.to(device) 
        x = x.view(-1, self.sequence_length, self.input_size)
        x = x.permute(0, 2, 1)
        out = [conv(x) for conv in self.convs]
        out = torch.cat(out, dim=1)
        out = out.view(-1, out.size(1))
        out = self.regression(out)
        return out