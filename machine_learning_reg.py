import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
#from sklearn.externals import joblib
from tqdm import tqdm ,trange
from tqdm.notebook import tnrange, tqdm_notebook
from sklearn.pipeline import make_pipeline
from datetime import datetime , timedelta
#from sklearn.externals import joblib
from config import DefaultConfig , switch
import torch
import time
import pickle
from sklearn.linear_model import Lasso , Ridge
from data import tem_spa_Time_series
from sklearn.ensemble import RandomForestRegressor
from utils import plain_evl_result
import fire
import inspect
import joblib

class Machine_learning_Regression():
    
    def __init__(self,pipe_lr,opt):
        self.pipe_lr = pipe_lr
        
        self.opt = opt
    
    def opt_update(self,**kwargs):
        
        self.opt._parse(kwargs)
        
    def time_rolling(self,train_date,dataframe,start_date,end_date,previous):
        for i in range(self.opt.input_size):
            tmp = dataframe.loc[train_date:,:]
            tmp = tmp.iloc[:,i].to_frame()
            if i == 0:
                
                time_previous=np.arange(previous)
                index = np.add.outer(time_previous,np.arange(tmp.shape[0]-previous)).transpose().reshape(-1)
                res = tmp.iloc[index]
                new = res.values.reshape(-1,previous)
            
            else:
                #tmp = dataframe.loc[self.train_date:,col].to_frame()
                
                time_previous=np.arange(previous)
                index = np.add.outer(time_previous,np.arange(tmp.shape[0]-previous)).transpose().reshape(-1)
                res = tmp.iloc[index]
                tmp_new = res.values.reshape(-1,previous)
                new = np.concatenate((new,tmp_new),axis=1)
        return new
    
    def train(self):
#         self.opt._parse(kwargs)
#         self.get_model_type()
        self.dirs_remake(self.opt.target_foler,self.opt.name)
        #self.dirs_remake(self.opt.target_foler,self.opt.model_by_day)
        
        self.df = pd.read_csv(self.opt.dataframe_csv) 
#         self.df.iloc[:,2] = self.df.iloc[:,2].map(format)
#         format = lambda x : '%d' %x
        self.df['bias'] = self.df['bias'].astype(int)
        self.evl_df = pd.DataFrame(columns=['Date','device_ID','MSE','R2_score','bias'])
        for i in trange(self.df.shape[0], desc='progressing device number'):
        #for i in range(self.df.shape[0]):
           
            #df_1 = pd.DataFrame(columns=['Date','MSE','Score'])
            if os.path.exists(os.path.join(self.opt.train_data_root, str(self.df.iloc[i]['device_ID'])+'.csv')):
                pm2_5 = pd.read_csv(os.path.join(self.opt.train_data_root, str(self.df.iloc[i]
                                   ['device_ID'])+'.csv'),index_col=0,parse_dates=True)
            else:
                continue
                
            if self.opt.load_model_path:
                try:
                    for dirPath, dirNames, fileNames in os.walk(
                                os.path.join('save',
                                             self.opt.load_model_path,str(self.df.iloc[i]['device_ID']))):
                        self.load(os.path.join(dirPath,fileNames[0]))
                        
                except:
                    path = os.path.join('save',
                                        self.opt.load_model_path,str(self.df.iloc[i]['device_ID']))
                    raise ValueError(f'Not exist {path!r} model path!')
                    
            end = datetime.strptime(self.df.iloc[i]['time'][:-3],"%Y-%m-%d %H:%M")
            end_previous = str(end + timedelta(minutes = -1 ))
            
            self.opt_update(end_dates = end_previous)
            select_pm_2_5_x = pm2_5.loc[:self.opt.end_dates]

            start = datetime.strptime(self.opt.start_dates,"%Y-%m-%d %H:%M:%S")
            train_date = str(start + timedelta(minutes= (-self.opt.previous) ))
            new_data = self.time_rolling(train_date,
                    select_pm_2_5_x,self.opt.start_dates,self.opt.end_dates,self.opt.previous)
           
            labels = select_pm_2_5_x.loc[self.opt.start_dates:]
            labels = labels.loc[:,['label']].values
            
            if self.opt.model_train:
                self.pipe_lr.fit(new_data, labels)
                if self.opt.model_save:
                    self.save(self.opt.name,str(self.df.iloc[i]['device_ID']))
            #end = str(end)[:-3]
            if self.opt.model_test:
                self.predict(pm2_5,end,'%Y-%m-%d',self.opt.name,i)
        if self.opt.model_test:
            self.evl_df.to_csv(os.path.join
                   (self.opt.target_foler,self.opt.name,"evl_df.csv"),index=0)
            plain_evl_result(self.evl_df,self.opt.target_foler,self.opt.name,self.opt.at_n)
        
    def load(self,path):
        self.pipe_lr = joblib.load(path)
        
                    
    def dirs_remake(self,target_foler,model):
        if os.path.exists(os.path.join(target_foler,model)):   #如果存在資料夾 , 刪除並重建一個
            shutil.rmtree(os.path.join(target_foler,model))
            os.mkdir(os.path.join(target_foler,model))
        else:
            os.mkdir(os.path.join(target_foler,model))
            
    def predict(self,pm2_5,date,day_format,target,index):
        #df_1 = pd.DataFrame(columns=['Date','MSE','Score'])
        #for date in dates: 
        test_y , pm2_5_y_pred = self.data_output(pm2_5,date,day_format)
        self.evl_df = self.evl_df.append({"Date":date.strftime(day_format),
                              "device_ID":self.df.iloc[index]['device_ID'],
                              "MSE":mean_squared_error(test_y, pm2_5_y_pred)
                        ,"R2_score":r2_score(test_y, pm2_5_y_pred),
                      "bias": self.df.iloc[index]['bias']},ignore_index=True)
    #df_1.to_csv("Linear_pm_2_5/"+str(df.iloc[i]['deviceId']) + ".csv",index=0)
        #df_1.to_csv(os.path.join(self.opt.target_foler,target,str(self.df.iloc[index]['device_ID']) + ".csv"),index=0)
    
    def data_output(self,pm2_5,date,day_format):
        #print(date.strftime(day_format))
        test = pm2_5[date.strftime(day_format)]
        test_start_dates = str(test.index[0])
        test_end_dates = str(test.index[test.shape[0]-1])
        start = datetime.strptime(test_start_dates[:-3],"%Y-%m-%d %H:%M")
        test_train_date = str(start + timedelta(minutes= (-self.opt.previous) ))
        test = pm2_5.loc[:test_end_dates]
        new_data = self.time_rolling(test_train_date,test,test_start_dates,test_end_dates,self.opt.previous)
        #test = test.loc[test_start_dates:]
        labels = test.loc[test_start_dates:]
        labels = labels.loc[:,['label']].values
        pm2_5_y_pred = self.pipe_lr.predict(new_data)   
        pm2_5_y_pred = pm2_5_y_pred.reshape(-1, 1)
        labels = labels.reshape(-1, 1)
        return labels , pm2_5_y_pred
    
    def save(self,name, device):
        
        if not os.path.exists(os.path.join('.','save')):
            os.mkdir(os.path.join('.','save'))
        #檢查checkpoints目錄下是否有模型的資料夾存在
        if not os.path.exists(os.path.join('save',name)):
            os.mkdir(os.path.join('save',name))
        if not os.path.exists(os.path.join('save',name,device)):
            os.mkdir(os.path.join('save',name,device))
        prefix = './save/' + name + '/' + device + '/'
            
        #names = time.strftime(prefix + '%Y_%m%d_%H:%M:%S.pkl')
        names = (prefix +self.opt.name+'.pkl')
        joblib.dump(self.pipe_lr, names)
        return names
    
def regression(**kwargs):
    opt = DefaultConfig()
    opt._parse(kwargs)
    model_kwargs = {}
    for case in switch(opt.model):
        if case('Lasso'):
            for k in inspect.getfullargspec(Lasso).args:
                if hasattr(opt, k):
                    model_kwargs[k] = getattr(opt,k)
            pipe_lr = make_pipeline(Lasso(**model_kwargs))
            break
        if case ('Ridge'):
            for k in inspect.getfullargspec(Ridge).args:
                if hasattr(opt, k):
                    model_kwargs[k] = getattr(opt,k)
            pipe_lr = make_pipeline(Ridge(**model_kwargs))
            break

        if case ('RandomForest'):
            for k in inspect.getfullargspec(RandomForestRegressor).args:
                if hasattr(opt, k):
                    model_kwargs[k] = getattr(opt,k)
            pipe_lr = make_pipeline(RandomForestRegressor(**model_kwargs))
            break
        if case():
            raise ValueError(f'Invalid inputs model type ,must be'
                                '{"Lasso"!r} or {"Ridge"!r} or {"RandomForest"!r} !')
            
    Regression = Machine_learning_Regression(pipe_lr,opt)
    Regression.train()
    
if __name__ == '__main__':
    fire.Fire()