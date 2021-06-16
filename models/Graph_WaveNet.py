import torch
import torch.nn as nn
from torch.nn import BatchNorm2d, Conv1d, Conv2d, ModuleList, Parameter
import torch.nn.functional as F
from .basic_module import BasicModule


class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncvl,vw->ncwl',(x,A))
        return x.contiguous()

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        """
        其实这就是一个裁剪的模块，裁剪多出来的padding
        """
        return x[:, :, : , :-self.chomp_size].contiguous()

class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)

class gcn(nn.Module):
    def __init__(self,c_in,c_out,dropout,support_len=3,order=2):
        super(gcn,self).__init__()
        self.nconv = nconv()
        c_in = (order*support_len+1)*c_in
        self.mlp = linear(c_in,c_out)
        self.dropout = dropout
        self.order = order

    def forward(self,x,support):
        out = [x]
        for a in support:
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class gwnet(BasicModule):
    def __init__(self,device=torch.device('cpu'), num_nodes = 6, dropout=0.3, gcn_bool=True, addaptadj=True, aptinit=None, 
             in_dim=1,out_dim=1,sequence_length = 30,residual_channels=32,dilation_channels=32,skip_channels=256,
                 end_channels=512,kernel_size=2,blocks=4,layers=2):
        super().__init__()
        
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.seq_len = sequence_length
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj
        self.num_nodes = num_nodes
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()            # Batch Normalize list
        self.gconv = nn.ModuleList()
        self.Chomp = nn.ModuleList()
        self.kernel_size = kernel_size
        self.dilation_channels = dilation_channels
        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.end_channels = end_channels
        self.out_dim = out_dim
        self.aptinit = aptinit
        self.device = device
        self.receptive_field = 1
    
    def set_Lk(self,Lk):
        if self.gcn_bool and self.supports is not None:
            if self.addaptadj:
                self.new_supports[0] = Lk
            else:
                self.supports[0] = Lk
    
    def set_network(self,Lk=None):
        self.supports = Lk
        self.supports_len = 0  # Number of Adjacency Matrix
        
        if self.supports is not None:
            self.supports_len += len(self.supports)
            
        if self.gcn_bool and self.addaptadj:
            if self.aptinit is None:
                if self.supports is None:
                    self.supports = []
                # Randomly initialize source node embedding and target node embedding
                self.nodevec1 = nn.Parameter(torch.randn(self.num_nodes, 10).to(self.device), 
                                             requires_grad=True).to(self.device)
                self.nodevec2 = nn.Parameter(torch.randn(10, self.num_nodes).to(self.device), 
                                             requires_grad=True).to(self.device)
                self.supports_len +=1
            else:
                # Initialize source node embedding and target node embedding with aptinit
                if self.supports is None:
                    self.supports = []
                m, p, n = torch.svd(aptinit)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(self.device)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(self.device)
                self.supports_len += 1
                
                
        for b in range(self.blocks):
            additional_scope = self.kernel_size - 1
            new_dilation = 1
            for i in range(self.layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=self.residual_channels,
                                                   out_channels=self.dilation_channels,
                                                   kernel_size=(1,self.kernel_size),dilation=new_dilation,
                                                  padding=(0,(self.kernel_size-1) * new_dilation)))
                
                self.gate_convs.append(nn.Conv1d(in_channels=self.residual_channels,
                                                 out_channels=self.dilation_channels,
                                                 kernel_size=(1, self.kernel_size), dilation=new_dilation,
                                                padding=(0,(self.kernel_size-1) * new_dilation)))
                self.Chomp.append(Chomp1d((self.kernel_size-1) * new_dilation))
                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv1d(in_channels=self.dilation_channels,
                                                     out_channels=self.residual_channels,
                                                     kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv1d(in_channels=self.dilation_channels,
                                                 out_channels=self.skip_channels,
                                                 kernel_size=(1, 1)))
                
                self.bn.append(nn.BatchNorm2d(self.residual_channels))  # Batch Normalize 
                new_dilation *=2
                self.receptive_field += additional_scope
                additional_scope *= 2
                if self.gcn_bool:
                    self.gconv.append(gcn(self.dilation_channels,self.residual_channels,
                                          self.dropout,support_len=self.supports_len))
                    
        # End convolution
        self.end_conv_1 = nn.Conv2d(in_channels=self.skip_channels,
                                  out_channels=self.end_channels,
                                  kernel_size=(1,1),
                                  bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=self.end_channels,
                                    out_channels=self.out_dim,
                                    kernel_size=(1,self.seq_len),
                                    bias=True)
        
        #self.end_conv_1 = nn.Linear(self.skip_channels * self.seq_len, self.end_channels)

        #self.end_conv_2 = nn.Linear(self.end_channels, self.out_dim)
        
        
    def forward(self, input,device):
        input = input.view(-1, 1,self.num_nodes, self.seq_len)
        
        in_len = input.size(3)
        
        '''
        if in_len<self.receptive_field:
            x = nn.functional.pad(input,(self.receptive_field-in_len,0,0,0))
        else:
            x = input
        '''
        x = input
        x = self.start_conv(x)
        skip = 0

        # calculate the current adaptive adj matrix once per iteration
        self.new_supports = None
        if self.gcn_bool and self.addaptadj and self.supports is not None:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            self.new_supports = self.supports + [adp]
            
        #print("adj matrix: ",new_supports)
        # WaveNet layers
        for i in range(self.blocks * self.layers):

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*

            #(dilation, init_dilation) = self.dilations[i]

            #residual = dilation_func(x, dilation, init_dilation, i)
            residual = x
            # dilated convolution
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            filter = self.Chomp[i](filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            gate = self.Chomp[i](gate)
            x = filter * gate

            # parametrized skip connection

            s = x
            s = self.skip_convs[i](s)
            try:
                #skip = skip[:, :, :,  -s.size(3):]
                skip = skip
            except:
                skip = 0
            skip = s + skip


            if self.gcn_bool and self.supports is not None:
                if self.addaptadj:
                    x = self.gconv[i](x, self.new_supports)
                else:
                    x = self.gconv[i](x,self.supports)
            else:
                x = self.residual_convs[i](x)

            #x = x + residual[:, :, :, -x.size(3):]

            x = x + residual

            x = self.bn[i](x)

        x = F.relu(skip)
        #x = x[:,:,0,-(self.seq_len):]

        #x = x.reshape(x.size(0), -1)
        x = F.relu(self.end_conv_1(x))

        x = self.end_conv_2(x)
        return x
        