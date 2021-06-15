# coding:utf8
from torch import nn
from .basic_module import BasicModule
from torch.autograd import Variable
import torch

class LSTM(BasicModule):
    def __init__(self, input_size, hidden_size, layers, num_classes,sequence_length):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = layers
        self.lstm = nn.LSTM(input_size, hidden_size, layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        '''
        self.regression = nn.Sequential()
        self.regression.add_module('linear0',nn.Linear(6, hidden[0]))
        self.regression.add_module('rule0', nn.ReLU())
        for i , hid in enumerate(hidden):
            if i == len(hidden) - 1:
                self.regression.add_module(('linear'+str(i+1)) , nn.Linear(hidden[i], 1 ))
            else:
                self.regression.add_module(('linear'+str(i+1)), nn.Linear(hidden[i], hidden[i+1]))
                self.regression.add_module(('relu' + str(i+1)), nn.ReLU())
        '''
    def forward(self, input_seq,device):
        # Set initial states
        #spa = input_seq[:,0:5]
        #input_seq = input_seq[:,5:]
        #input_seq = input_seq.view(-1, self.sequence_length, self.input_size)
        input_seq = input_seq.view(-1, self.input_size , self.sequence_length )
        input_seq = input_seq.permute(0,2,1)
        h0 = torch.zeros(self.num_layers, input_seq.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, input_seq.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(input_seq, (h0, c0))  
        
        # Decode hidden state of last time step
        out = self.fc(out[:, -1, :])  
        #inputs = torch.cat([out,spa],dim=1) 
        #out = self.regression(inputs)
        return out
