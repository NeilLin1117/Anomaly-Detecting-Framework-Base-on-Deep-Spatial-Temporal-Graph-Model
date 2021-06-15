# coding:utf8
from torch import nn
from .basic_module import BasicModule
import torch


class CNN_LSTM(BasicModule):
    def __init__(self,in_channels,out_channels,hidden_size,num_layers,
                 out_feature,input_size,lstm_sequence,sequence_length,hidden,window_sizes):
        super(CNN_LSTM, self).__init__()
        self.model_name = 'CNN_LSTM'
        self.convs = nn.ModuleList([
                nn.Sequential(nn.Conv1d(in_channels = in_channels, 
                                        out_channels = out_channels, 
                                        kernel_size=h),
                              nn.ReLU(),
                              nn.MaxPool1d(kernel_size=sequence_length-h+1))
                     for h in window_sizes
                ])
        self.in_channels = in_channels
        self.input_size = input_size
        self.sequence_length = sequence_length
        self.lstm_sequence = lstm_sequence
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.out_channels = out_channels
        self.window_sizes = window_sizes
        
        '''
        self.regression = nn.Sequential()
        self.regression.add_module('linear0',nn.Linear(6, hidden[0]))
        self.regression.add_module('relu0', nn.ReLU())
        for i , hid in enumerate(hidden):
            if i == len(hidden) - 1:
                self.regression.add_module(('linear'+str(i+1)) , nn.Linear(hidden[i], out_feature ))
            else:
                self.regression.add_module(('linear'+str(i+1)), nn.Linear(hidden[i], hidden[i+1]))
                self.regression.add_module(('relu' + str(i+1)), nn.ReLU())
        '''
        
    def forward(self,x,device):
        #x = x.to(device) 
        #spa = x[:,0:5]
        #x = x[:,5:]
        x = x.view(-1, self.sequence_length, self.in_channels)
        x = x.permute(0, 2, 1)
        out = [conv(x) for conv in self.convs]
        out = torch.cat(out, dim=1)
        out = out.view(-1, out.size(1))
        #input_seq = out.view(-1, self.out_channels*len(self.window_sizes), self.input_size)
        input_seq = out.view(-1, self.lstm_sequence, self.input_size)
        h0 = torch.zeros(self.num_layers, input_seq.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, input_seq.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(input_seq, (h0, c0))
        
        return out