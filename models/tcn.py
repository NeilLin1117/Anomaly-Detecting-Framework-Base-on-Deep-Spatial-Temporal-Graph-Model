from torch import nn
from models import TemporalConvNet
from .basic_module import BasicModule


class TCN(BasicModule):
    def __init__(self, sequence_length,input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        setattr(self, 'seq_len', sequence_length)
        setattr(self, 'input_size', input_size)
        setattr(self, 'tcn', TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout))
        last_channel = num_channels[-1]
        setattr(self,'linear',nn.Linear(last_channel, output_size))
#         self.seq_len = sequence_length
#         self.input_size = input_size
#         self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
#         self.linear = nn.Linear(num_channels[-1], output_size)
   
    
    def forward(self, inputs,device):
        """Inputs have to have dimension (N, C_in, L_in)"""
        inputs = inputs.view(-1,self.input_size,self.seq_len)
        y1 = self.tcn(inputs)  # input should have dimension (N, C, L)
        y2 = y1[:, :, -1]
        o = self.linear(y2)
        return o