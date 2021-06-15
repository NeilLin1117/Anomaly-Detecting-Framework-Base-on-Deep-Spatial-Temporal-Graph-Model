# coding:utf8
from torch import nn
from .basic_module import BasicModule


class DNN(BasicModule):
    def __init__(self,input_size=6,sequence_length=30,hidden=[300,100,20],output_size = 1,dropout=0.2):
        super(DNN, self).__init__()
        self.model_name = 'DNN'
        self.regression = nn.Sequential()
        self.regression.add_module('linear0',nn.Linear(input_size*sequence_length, hidden[0]))
        self.regression.add_module('relu0', nn.ReLU())
        #self.regression.add_module('dropout0', nn.Dropout(dropout))
        for i , hid in enumerate(hidden):
            if i == len(hidden) - 1:
                self.regression.add_module(('linear'+str(i+1)) , nn.Linear(hidden[i], output_size ))
            else:
                self.regression.add_module(('linear'+str(i+1)), nn.Linear(hidden[i], hidden[i+1]))
                self.regression.add_module(('relu' + str(i+1)), nn.ReLU())
                #self.regression.add_module('dropout'+str(i+1), nn.Dropout(dropout))
        '''
        self.regression = nn.Sequential(
            nn.Linear(143, 300),
            nn.ReLU(),
            nn.Linear(300, 100),
            nn.ReLU(),
            nn.Linear(100, 20),
            nn.ReLU(),
            nn.Linear(20, 1)
        )
        '''
        
    def forward(self,x,device):
        #x = x.to(device) 
        x = self.regression(x)
        return x