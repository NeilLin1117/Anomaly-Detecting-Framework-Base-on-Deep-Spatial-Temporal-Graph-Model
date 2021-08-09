#coding:utf8
import torch as t
import torch.nn as nn
import time
import os


class BasicModule(nn.Module):
    """
    封装了nn.Module,主要是提供了save和load兩個方法
    """

    def __init__(self):
        super(BasicModule,self).__init__()
        self.model_name=str(type(self))# 默認名字

    def load(self, path):
        """
        可加載指定路徑的模型
        """
        self.load_state_dict(t.load(path))
        #weight = os.path.join(weight_path)
        #chkpt = t.load(path)
        #self.load_state_dict(chkpt['state_dict'])

    def save(self,name, device):
        """
        保存模型，默認使用“模型名字+時間”作為文件名
        如AlexNet_0710_23:57:29.pth
        """
        if not os.path.exists(os.path.join('.','save')):
            os.mkdir(os.path.join('.','save'))
            
        if name is None:
            #檢查checkpoints目錄下是否有模型的資料夾存在
            if device is not None:
                if not os.path.exists(os.path.join('save',self.model_name)):
                    os.mkdir(os.path.join('save',self.model_name))
                if not os.path.exists(os.path.join('save',self.model_name,device)):
                    os.mkdir(os.path.join('save',self.model_name,device))
                prefix = 'save' + '/' + self.model_name + '/'+ device + '/'
            else:
                if not os.path.exists(os.path.join('save',self.model_name)):
                    os.mkdir(os.path.join('save',self.model_name))
                prefix = 'save' + '/' + self.model_name + '/'

        else:
            #檢查checkpoints目錄下是否有模型的資料夾存在
            if device is not None:
                if not os.path.exists(os.path.join('save',name)):
                    os.mkdir(os.path.join('save',name))
                if not os.path.exists(os.path.join('save',name,device)):
                    os.mkdir(os.path.join('save',name,device))
                prefix = 'save' + '/' + name + '/' + device + '/'
            else:
                if not os.path.exists(os.path.join('save',name)):
                    os.mkdir(os.path.join('save',name))
                prefix = 'save' + '/' + name + '/' 
        names = time.strftime(prefix + name +'.pth')
        t.save(self.state_dict(), names)
        return names
    
    @classmethod
    def generate_model(cls, config):
        return cls(**config)
    
    def get_optimizer(self, lr, weight_decay):
        return t.optim.RMSprop(self.parameters(), lr=lr, weight_decay=weight_decay)
        #return t.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
    
    def get_LR_Scheduler(self,optimizer,step_size,gamma):
        return t.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        