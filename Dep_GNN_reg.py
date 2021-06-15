import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.notebook import tnrange, tqdm_notebook
from tqdm import trange
from datetime import datetime , timedelta
from config import DefaultConfig , switch
import torch
import copy
import torch.nn as nn
import torch as t
import time
import inspect
from models import gwnet,STGCN
from data import tem_spa_Time_series
import torch.nn as nn
from utils import plain_evl_result
import fire
from sklearn.metrics import mean_squared_error, r2_score

class Dep_GNN_Regression():
    def __init__(self,model,opt):
        #self.opt = DefaultConfig()
        self.model = copy.deepcopy(model)
        self.opt = copy.deepcopy(opt)
    def opt_update(self,**kwargs):
        self.opt._parse(kwargs)
        
    def cheb_poly(self,L, Ks):
        n = L.shape[0]
        LL = [np.eye(n), L[:]]
        for i in range(2, Ks):
            LL.append(np.matmul(2 * L, LL[-1]) - LL[-2])
        return np.asarray(LL)
    
    def train(self):
#         self.opt._parse(kwargs)
        
        ########  get model type   ######
        #self.get_model_type()
        
        ####### begin train  #####
        self.dirs_remake(self.opt.target_foler,self.opt.name)
        
        self.df = pd.read_csv(self.opt.dataframe_csv)
        file_list = self.df['device_ID'].values.tolist()
#         if not os.path.exists(os.path.join(self.opt.target_foler,self.opt.name)):
#             os.mkdir(os.path.join(self.opt.target_foler,self.opt.name))
            
#         format = lambda x : '%d' %x
#         self.df.iloc[:,2] = self.df.iloc[:,2].map(format)
        
        self.evl_df = pd.DataFrame(columns=['Date','MSE','R2_score','bias'])
        for i in trange(self.df.shape[0], desc='1st loop'):
            loss_function = nn.MSELoss()
            lr = self.opt.lr
            
            if not os.path.exists(os.path.join(self.opt.train_data_root,
                                               str(self.df.iloc[i]['device_ID'])+'.csv')):
                continue
                
            end = datetime.strptime(self.df.iloc[i]['time'],"%Y-%m-%d %H:%M:%S")
            if str(end) == str(self.opt.start_dates):
                flag = 0
                
                pm2_5 = pd.read_csv(os.path.join(self.opt.train_data_root, 
                        str(self.df.iloc[i]['device_ID'])+'.csv'),index_col=0,parse_dates=True)
                L = np.load(os.path.join(self.opt.laplacian_folder, 
                                         str(self.df.iloc[i]['device_ID'])+'.npy'))
                # Graph WaveNet
                if self.opt.model == "gwnet":
                    Lk = [torch.Tensor(L.astype(np.float32)).to(self.opt.device)]
                
                # STGCN
                else:
                    Lk = self.cheb_poly(L, self.opt.ks)
                    Lk = torch.Tensor(Lk.astype(np.float32)).to(self.opt.device)
                    
                
                for name, params in self.model.named_parameters():
                    if name.find("end_conv_1") != -1 or name.find("st_conv1") != -1:
                        flag = 1
                        break
                if flag:
                    self.model.set_Lk(Lk)
                else:    
                    self.model.set_network(Lk)
                    
                if self.opt.load_model_path:
                    try:
                        for dirPath, dirNames, fileNames in os.walk(
                                    os.path.join('save',self.opt.load_model_path)):
                            self.model.load(os.path.join(dirPath,fileNames[0]))
                    except:
                        path = os.path.join('save',self.opt.load_model_path)
                        raise ValueError(f'Not exist {path!r} model path!')
                    
                    self.opt.load_model_path = None
                    
                if self.opt.model_test:
                    self.predict(self.model,pm2_5,end,'%Y-%m-%d',self.opt.name,i)
                    continue
                    
            end_previous = str(end + timedelta(minutes = -1 ))
            #end_previous = end_previous.strftime("%Y-%m-%d")
            self.opt_update(end_dates = end_previous)
                
            #pm2_5_x = pm2_5.loc[:self.opt.end_dates]
            
            if i == 0:
                L = np.load(os.path.join(self.opt.laplacian_folder, 
                                             str(self.df.iloc[i]['device_ID'])+'.npy'))
                # GraphWaveNet
                if self.opt.model == "gwnet":
                    Lk = [torch.Tensor(L.astype(np.float32)).to(self.opt.device)]
                else:
                    Lk = self.cheb_poly(L, self.opt.ks)
                    Lk = torch.Tensor(Lk.astype(np.float32)).to(self.opt.device)
                self.model.set_network(Lk)
                
            self.model = self.model.to(self.opt.device)
            
            ############# begin train ##############
            if self.opt.model_train:
                
                optimizer = self.model.get_optimizer(lr, self.opt.weight_decay)
                ###########   Create dataloader dict    ###########
#                 dataloader_list = {}

#                 for j in range(len(file_list)):
#                     if os.path.exists(os.path.join(self.opt.train_data_root, 
#                                                        str(file_list[j])+'.csv')):
#                         pm2_5 = pd.read_csv(os.path.join(self.opt.train_data_root, 
#                                      str(file_list[j])+'.csv'),index_col=0,parse_dates=True)
#                     else:
#                         continue
#                     dataset = tem_spa_Time_series(pm2_5,self.opt.previous
#                                 ,self.opt.start_dates,self.opt.end_dates,
#                                       node_cnt = self.opt.input_size,)
#                     train_loader = torch.utils.data.DataLoader(dataset=dataset,
#                                                        batch_size=self.opt.batch_size, 
#                                                        shuffle=False,num_workers = 0)
#                     dataloader_list[j] = train_loader

                total = []

                for t in trange((self.opt.max_epoch), desc='2nd loop', leave=False):    
                    total_loss = 0
                    for k, key in enumerate(file_list):
                        if os.path.exists(os.path.join(self.opt.train_data_root, 
                                                           str(key)+'.csv')):
                            pm2_5 = pd.read_csv(os.path.join(self.opt.train_data_root, 
                                         str(key)+'.csv'),index_col=0,parse_dates=True)
                        else:
                            continue
                        dataset = tem_spa_Time_series(pm2_5,self.opt.previous
                                    ,self.opt.start_dates,self.opt.end_dates,
                                          node_cnt = self.opt.input_size,)
                        train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                           batch_size=self.opt.batch_size, 
                                                           shuffle=False,num_workers = 0)  
#                     for k , (key, train_loader) in enumerate(dataloader_list.items()):
                        L = np.load(os.path.join(self.opt.laplacian_folder, 
                                                 str(key)+'.npy'))
                        # GraphWaveNet
                        if self.opt.model == "gwnet":
                            Lk = torch.Tensor(L.astype(np.float32)).to(self.opt.device)
                        # STGCN
                        else:
                            Lk = self.cheb_poly(L, self.opt.ks)
                            Lk = torch.Tensor(Lk.astype(np.float32)).to(self.opt.device)
                        if not (i == 0 and t == 0 and k == 0):
                            # Graph WaveNet
                            self.model.set_Lk(Lk)

                        if self.opt.load_model_path:
                            try:
                                for dirPath, dirNames, fileNames in os.walk(
                                            os.path.join('check',self.opt.load_model_path)):
                                    self.model.load(os.path.join(dirPath,fileNames[0]))
                            except:
                                path = path = os.path.join("check",self.opt.load_model_path)
                                raise ValueError(f'Not exist {path!r} model path!')
                            
                            self.opt.load_model_path = None
                        for s, (datas, labels) in enumerate(train_loader):
                            datas = datas.to(self.opt.device)
                            labels = labels.to(self.opt.device)
                            optimizer.zero_grad()
                            y_pred = self.model(datas,self.opt.device)

                            # Graph WaveNet loss function
                            if self.opt.model == "gwnet":
                                y_pred = y_pred[:,:,0,:].view(y_pred.size(0), -1)
                            #single_loss = loss_function(y_pred, labels)

                            #STGCN loss function
                            else:
                                y_pred = y_pred[:,:,:,0].view(y_pred.size(0), -1)  
                            single_loss = loss_function(y_pred, labels)

                            total_loss += single_loss.item()
                            single_loss.backward()
                            optimizer.step()

                    if self.opt.model_save:
                        self.model.save(self.opt.name,None)        

                    total.append(total_loss)

                if self.opt.visual:
                    plt.plot(range(self.opt.max_epoch), total)
                    plt.title(str(self.df.iloc[i]['device_ID']))
                    plt.ylabel('Cost')
                    plt.xlabel('Epochs')
                    plt.show()    
                    #scheduler.step()
                if self.opt.model_save:
                    self.model.save(self.opt.name,None)
            if self.opt.model_test:
                pm2_5 = pd.read_csv(os.path.join(self.opt.train_data_root, 
                             str(self.df.iloc[i]['device_ID'])+'.csv'),index_col=0,parse_dates=True)
                
                # Graph WaveNet
                #L = torch.Tensor(L.astype(np.float32)).to(self.opt.device)
                self.predict(self.model,pm2_5,end,'%Y-%m-%d',self.opt.name,i)
                
            self.opt.start_dates = str(end)
            
        if self.opt.model_test:
            self.evl_df.to_csv(os.path.join
                   (self.opt.target_foler,self.opt.name,"evl_df.csv"),index=0)
            plain_evl_result(self.evl_df,self.opt.target_foler,self.opt.name,self.opt.at_n)        
            
    def dirs_remake(self,target_foler,model):
        if os.path.exists(os.path.join(target_foler,model)):   #如果存在資料夾 , 刪除並重建一個
            shutil.rmtree(os.path.join(target_foler,model))
            os.mkdir(os.path.join(target_foler,model))
        else:
            os.mkdir(os.path.join(target_foler,model))
            
    def data_output(self,Model,pm2_5,date,day_format,index):
        test = pm2_5[date.strftime(day_format)]
        test_start_dates = str(test.index[0])
        test_end_dates = str(test.index[test.shape[0]-1])
        L = np.load(os.path.join(self.opt.laplacian_folder, 
                                 str(self.df.iloc[index]['device_ID'])+'.npy'))
        if self.opt.model == "gwnet":
            Lk = [torch.Tensor(L.astype(np.float32)).to(self.opt.device)]
        else:
            Lk = self.cheb_poly(L, self.opt.ks)
            Lk = torch.Tensor(Lk.astype(np.float32)).to(self.opt.device)
        Model.set_Lk(Lk)
        dataset = tem_spa_Time_series(pm2_5,self.opt.previous,
                                  test_start_dates,test_end_dates,node_cnt = self.opt.input_size,
                                      )
        test_loader = torch.utils.data.DataLoader(dataset=dataset,
                                           batch_size=self.opt.batch_size, 
                                           shuffle=False)
        with torch.no_grad():
            for k ,(datas, labels) in enumerate(test_loader): 
                Model = Model.to(self.opt.device)
                datas = datas.to(self.opt.device)
                outputs = Model(datas,self.opt.device)
                
                # GraphWaveNet Output
                if self.opt.model == "gwnet":
                    outputs = outputs[:,:,0,:].view(outputs.size(0), -1)
                # STGCN Output
                else:
                    outputs = outputs[:,:,:,0].view(outputs.size(0), -1)  
                if k == 0:
                    result = outputs.cpu().numpy()
                    labs = labels.cpu().numpy()
                else:
                    tmp = outputs.cpu().numpy()
                    tmp_labs = labels.cpu().numpy()
                    #tmp_labs = tmp_labs.reshape(-1, 1)
                    result = np.concatenate([result, tmp], axis=0)
                    labs = np.concatenate([labs, tmp_labs], axis=0)
        return labs , result
    
    def predict(self,Model,pm2_5,date,day_format,target,index):
        #df_1 = pd.DataFrame(columns=['Date','MSE','Score'])
        #for date in dates:
        labs , result = self.data_output(Model,pm2_5,date,day_format,index)
        
        self.evl_df = self.evl_df.append({"Date":date.strftime(day_format),
                            "MSE":mean_squared_error(labs, result)
                                ,"R2_score":r2_score(labs, result),
                                 "bias": self.df.iloc[index]['bias']},ignore_index=True)
        
def regression(**kwargs):
    opt = DefaultConfig()
    opt._parse(kwargs)
    model_kwargs = {}
    for case in switch(opt.model):
        if case('gwnet'):
            for k in inspect.getfullargspec(gwnet).args:
                if hasattr(opt, k):
                    model_kwargs[k] = getattr(opt,k)
            model = gwnet(**model_kwargs)
            break
        if case ('STGCN'):
            for k in inspect.getfullargspec(STGCN).args:
                if hasattr(opt, k):
                    model_kwargs[k] = getattr(opt,k)
            model = STGCN(**model_kwargs)
            break  

        if case():
            raise ValueError(f'Invalid inputs model type ,must be {"gwnet"!r} or {"STGCN"!r} !')
    Regression = Dep_GNN_Regression(model,opt)
    Regression.train()
    
if __name__ == '__main__':
    fire.Fire()