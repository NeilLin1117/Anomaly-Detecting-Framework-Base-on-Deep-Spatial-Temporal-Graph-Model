# coding:utf8
import torch
from torch.utils import data
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime , timedelta

class tem_spa_Time_series(data.Dataset):
    
    def __init__(self,select_pm_2_5,previous,start_date,end_date,*,node_cnt = 6 , mul_label = False):
        #scaler_x = MinMaxScaler(feature_range=(-1, 1))
        #scaler_y = MinMaxScaler(feature_range=(-1, 1))
        self.node_cnt = node_cnt
        select_pm_2_5 = select_pm_2_5.loc[:end_date]
        self.pm_2_5 = select_pm_2_5
        # '2018-01-01 01:00'  '2018-01-01 00:30'  '2018-12-31 23:59'
        start = datetime.strptime(start_date,"%Y-%m-%d %H:%M:%S")
        self.train_date = str(start + timedelta(minutes= (-previous) ))
        self.start_date = start_date
        self.end_date = end_date
        
        self.datas = self.time_rolling(select_pm_2_5,previous)
        test = self.pm_2_5.loc[self.start_date:]
        #print(self.datas.shape)
        #new_data = pd.merge(test,new_data,left_index=True,right_index=True)
        #pm2_5_x = (new_data.iloc[:,1:].values)
        #pm2_5_y = (new_data.loc[:,['predict']].values)
        #spatio = (new_data.iloc[:,[1,2]].values)
        #temporal = (new_data.loc[:,[previous-2,previous-1]].values)
        self.datas = torch.FloatTensor(self.datas)
        if mul_label:
            self.labels = torch.FloatTensor(test.values)
        else:
            self.labels = torch.FloatTensor(test.loc[:,['label']].values)
        #self.spatio = torch.FloatTensor(test.iloc[:,[1,2]].values)
        #self.temporal = torch.FloatTensor(temporal.loc[:,[previous-2,previous-1]].values)

        #if visual:
        #    ht = Heatmap(test['predict'].to_frame(),temporal,previous)
        #    ht.plots(folder)
        
    def time_rolling(self,dataframe,previous):
        #for i , col in enumerate(dataframe.columns):
        for i in range(self.node_cnt):
            tmp = dataframe.loc[self.train_date:,:]
            tmp = tmp.iloc[:,i].to_frame()
            
            if i == 0:
                #tmp = dataframe.loc[self.train_date:,col].to_frame()
                
                time_previous=np.arange(previous)
                index = np.add.outer(time_previous,np.arange(tmp.shape[0]-previous)).transpose().reshape(-1)
                res = tmp.iloc[index]
                new = res.values.reshape(-1,previous)
                #minutes = pd.date_range(self.start_date,self.end_date,freq='T')
                #temporal = pd.DataFrame(ss,index = minutes )
                #new = pd.DataFrame(ss,index = minutes )
            else:
                #tmp = dataframe.loc[self.train_date:,col].to_frame()
                
                time_previous=np.arange(previous)
                index = np.add.outer(time_previous,np.arange(tmp.shape[0]-previous)).transpose().reshape(-1)
                res = tmp.iloc[index]
                tmp_new = res.values.reshape(-1,previous)
                #minutes = pd.date_range(self.start_date,self.end_date,freq='T')
                #tmp_new = pd.DataFrame(ss,index = minutes )
                #new = pd.merge(new,tmp_new,left_index=True,right_index=True)
                new = np.concatenate((new,tmp_new),axis=1)
        return  new
    
    def __getitem__(self,index):  #回傳資料集的index資料
        return self.datas[index],self.labels[index] 
    
    def __len__(self):
        return self.pm_2_5.loc[self.start_date:].shape[0]  #回傳資料集大小