from torch import nn
from models import TemporalConvNet
from .basic_module import BasicModule


class TCN(BasicModule):
    def __init__(self,sequence_length=30,input_size=6, output_size=1, num_channels=[16,32,48], kernel_size=3, dropout=0.0):
        super(TCN, self).__init__()
        self.seq_len = sequence_length
        self.input_size = input_size
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, inputs,device):
        """Inputs have to have dimension (N, C_in, L_in)"""
        inputs = inputs.view(-1,self.input_size,self.seq_len)
        y1 = self.tcn(inputs)  # input should have dimension (N, C, L)
        o = self.linear(y1[:, :, -1])
        return o