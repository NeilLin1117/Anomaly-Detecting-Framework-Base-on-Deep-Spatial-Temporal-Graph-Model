import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from .basic_module import BasicModule

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        """
        其实这就是一个裁剪的模块，裁剪多出来的padding
        """
        return x[:, :, : , :-self.chomp_size].contiguous()

class align(nn.Module):
    def __init__(self, c_in, c_out):
        super(align, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        if c_in > c_out:
            self.conv1x1 = nn.Conv2d(c_in, c_out, 1)

    def forward(self, x):
        if self.c_in > self.c_out:
            return self.conv1x1(x)
        if self.c_in < self.c_out:
            return F.pad(x, [0, 0, 0, 0, 0, self.c_out - self.c_in, 0, 0])
        return x

class temporal_conv_layer(nn.Module):
    def __init__(self, kt, c_in, c_out, act="relu"):
        super(temporal_conv_layer, self).__init__()
        self.kt = kt
        self.act = act
        self.c_out = c_out
        self.align = align(c_in, c_out)
        if self.act == "GLU":
            self.conv = nn.Conv2d(c_in, c_out * 2, (kt, 1), 1)
        else:
            self.conv = nn.Conv2d(c_in, c_out, (kt, 1), 1)

    def forward(self, x):
        x_in = self.align(x)[:, :, self.kt - 1:, :]
        if self.act == "GLU":
            x_conv = self.conv(x)
            return (x_conv[:, :self.c_out, :, :] + x_in) * torch.sigmoid(x_conv[:, self.c_out:, :, :])
        if self.act == "sigmoid":
            return torch.sigmoid(self.conv(x) + x_in)
        return torch.relu(self.conv(x) + x_in)

class spatio_conv_layer(nn.Module):
    def __init__(self, ks, c, Lk):
        super(spatio_conv_layer, self).__init__()
        self.Lk = Lk
        self.theta = nn.Parameter(torch.FloatTensor(c, c, ks))
        self.b = nn.Parameter(torch.FloatTensor(1, c, 1, 1))
        self.reset_parameters()

    def set_Lk(self,Lk):
        self.Lk = Lk

    def reset_parameters(self):
        init.kaiming_uniform_(self.theta, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.theta)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.b, -bound, bound)

    def forward(self, x):
        x_c = torch.einsum("knm,bitm->bitkn", self.Lk, x)
        x_gc = torch.einsum("iok,bitkn->botn", self.theta, x_c) + self.b
        return torch.relu(x_gc + x)

class st_conv_block(nn.Module):
    def __init__(self, ks, kt, n, c, p, Lk):
        super(st_conv_block, self).__init__()
        self.tconv1 = temporal_conv_layer(kt, c[0], c[1], "GLU")
        self.sconv = spatio_conv_layer(ks, c[1], Lk)
        self.tconv2 = temporal_conv_layer(kt, c[1], c[2])
        self.ln = nn.LayerNorm([n, c[2]])
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        x_t1 = self.tconv1(x)
        x_s = self.sconv(x_t1)
        x_t2 = self.tconv2(x_s)
        x_ln = self.ln(x_t2.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)        
        return self.dropout(x_ln)

class fully_conv_layer(nn.Module):
    def __init__(self, c):
        super(fully_conv_layer, self).__init__()
        self.conv = nn.Conv2d(c, 1, 1)

    def forward(self, x):
        return self.conv(x)

class output_layer(nn.Module):
    def __init__(self, c, T, n ):
        super(output_layer, self).__init__()
        self.tconv1 = temporal_conv_layer(T, c, c, "GLU")
        self.ln = nn.LayerNorm([n, c])
        self.tconv2 = temporal_conv_layer(1, c, c, "sigmoid")
        self.fc = fully_conv_layer(c)

    def forward(self, x):
        x_t1 = self.tconv1(x)
        x_ln = self.ln(x_t1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x_t2 = self.tconv2(x_ln)
        x_t2 = self.fc(x_t2)
        return x_t2

class STGCN(BasicModule):
    def __init__(self, ks, kt, bs, sequence_length, num_nodes ,dropout ):
        super(STGCN, self).__init__()
        bs = np.array(bs).reshape(-1,3).tolist()
        self.ks, self.kt, self.bs, self.T, self.n, self.p = ks, kt, bs, sequence_length, num_nodes, dropout
        #self.output = output_layer(self.bs[2][2], self.T - 6 * (self.kt - 1), self.n)
#         self.output = output_layer(self.bs[1][2], self.T - 4 * (self.kt - 1), self.n)
        self.output = output_layer(self.bs[len(bs)-1][2], self.T - 2 * len(bs) * (self.kt - 1), self.n)
        
    def set_network(self,Lk):
        self.st_conv_list = nn.ModuleList(
                [st_conv_block(self.ks, self.kt, self.n, bs , self.p, Lk) for bs in self.bs])
#         self.st_conv_list = [st_conv_block(self.ks, self.kt, self.n, bs , self.p, Lk) for bs in self.bs]
        #self.st_conv1 = st_conv_block(self.ks, self.kt, self.n, self.bs[0], self.p, Lk)
        #self.st_conv2 = st_conv_block(self.ks, self.kt, self.n, self.bs[1], self.p, Lk)
        #self.st_conv3 = st_conv_block(self.ks, self.kt, self.n, self.bs[2], self.p, Lk)

    def set_Lk(self,Lk):        
        for st_conv in self.st_conv_list:
            st_conv.sconv.set_Lk(Lk)
#         self.st_conv1.sconv.set_Lk(Lk)
#         self.st_conv2.sconv.set_Lk(Lk)

    def forward(self, x,device):
#         print(x.size())
        x = x.view(-1,1, self.n, self.T)
        x = x.permute(0,1,3,2)
        #x = x.view(-1, 1, self.T, self.n)
        for st_conv in self.st_conv_list:
            x = st_conv(x)
        #x_st1 = self.st_conv1(x)
        #x_st2 = self.st_conv2(x_st1)
        #x_st3 = self.st_conv3(x_st2)
        #print(x_st3.size())
        #end_channels = x_st2.size(1) * x_st2.size(2)
        return self.output(x)
