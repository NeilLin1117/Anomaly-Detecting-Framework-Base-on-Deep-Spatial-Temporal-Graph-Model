from torch import nn
import torch


class seq_tem_loss(nn.Module):
    
    def __init__(self):
        super(seq_tem_loss,self).__init__()
        
    def forward(self,x,y,spatio,temporal,use_gpu):
        
        loss = torch.mean(torch.pow((x - y), 2) / 2)
        
        return loss   