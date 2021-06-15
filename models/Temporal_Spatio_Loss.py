# coding:utf8
from torch import nn
import torch


class tem_spa_loss(nn.Module):
    
    def __init__(self):
        super(tem_spa_loss,self).__init__()
        
    def forward(self,x,y,spatio,temporal,use_gpu):
        if use_gpu:
            ct = torch.Tensor([20,10]).cuda()
            cs = torch.Tensor([10,20]).cuda()
        else:
            ct = torch.Tensor([20,10])
            cs = torch.Tensor([10,20])
        loss = torch.mean(torch.pow((x - y), 2) / 2)
        for i in range(int(spatio.shape[1])):
            loss += torch.mean(torch.pow((x - spatio[:,[i]]), 2) * torch.exp(-cs[i]) / 2)
        for i in range(int(temporal.shape[1])):   
            loss += torch.mean(torch.pow((x - temporal[:,[i]]), 2) * torch.exp(-ct[i]) / 2)
        return loss   