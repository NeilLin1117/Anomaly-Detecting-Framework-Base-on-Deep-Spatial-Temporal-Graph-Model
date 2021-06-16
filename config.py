# coding:utf8
import warnings
import pandas as pd
import torch as t

class DefaultConfig(object):
    name = 'test'  # 為模型取名稱
    model_by_day = 'LSTM_pm_2_5_by_day'
    train_data_root = './data/temporal_spatio_pm_2_5'  
    train_144_path = './data/temporal_spatio_pm_2_5_144.gz'
    laplacian_folder = './data/normalized_laplacian'
    laplacian_144_path = './data/normalized_laplacian_144.npy'
    dataframe_csv = './data/device_ground_truth.csv'
    target_foler = 'output'
    end_dates = '2018-05-31 23:59'
    start_dates = '2018-01-01 01:30:00'
    model = "gwnet"  #模型的型態
    visual = True
    model_train = True
    #test_data_root = './data/test1'  # 测试集存放路径
    load_model_path = None  # load pretrain model，If None represent not load pretrain model path
    batch_size = 64  # batch size
    use_gpu = True  # user GPU or not
    model_save = False
    num_workers = 4  # how many workers for loading data
    dropout = 0.0
    in_feature = 35
    out_feature = 1
    num_layer = 5
    num_classes = 1
    sequence_length = 30
    input_size = 6
    output_size = 1
    model_test = True
    previous = 30
    hidden = [300,100,20]
    hidden_size = 128
    num_channels = [16,32,48]
    max_epoch = 30
    lr = 0.001  # initial learning rate
    gamma = 0.7  # when val_loss increase, lr = lr*lr_decay
    step_size = 5
    at_n = 50
    #### Random Forest ####
    n_estimators = 25
    criterion = 'mse'
    random_state = None
    max_depth=None, 
    min_samples_split=2, 
    min_samples_leaf=1, 
    min_weight_fraction_leaf=0.0, 
    max_features='auto', 
    max_leaf_nodes=None, 
    min_impurity_decrease=0.0, 
    min_impurity_split=None, 
    bootstrap=True, 
    oob_score=False, 
    n_jobs=None,
    verbose=0, 
    warm_start=False, 
    ccp_alpha=0.0, 
    max_samples=None
    #### Lasso  and Ridge ####
    alpha=1.0
    
    fit_intercept=True, 
    normalize=False, 
    precompute=False, 
    copy_X=True, 
    max_iter=1000, 
    tol=0.0001, 
    warm_start=False, 
    positive=False,
    selection='cyclic
    solver='auto'
    
    #### STGCN ####
    num_nodes = 6
    ks, kt = 3, 3
    bs = [1, 32, 64, 64, 32, 128]
    
    #### GraphWaveNet Parameters ####
    gcn_bool = True
    addaptadj = True
    aptinit = None
    in_dim = 1 
    out_dim = 1
    residual_channels = 4
    dilation_channels = 4
    skip_channels = 8
    end_channels = 16
    kernel_size = 3
    blocks = 4
    layers = 3
    
    #weight_decay = 0e-5  # 損失函數
    weight_decay = 0
    device = t.device('cuda')
    def _parse(self, kwargs):
        """
        根據字典kwargs 更新 config参数
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)
        
        if opt.use_gpu:
            setattr(self, 'device', t.device('cuda'))
        else:
            setattr(self, 'device', t.device('cpu'))
        #opt.device =t.device('cuda') if opt.use_gpu else t.device('cpu')


        #print('user config:')
        #for k, v in self.__class__.__dict__.items():
            #if not k.startswith('_'):
                #print(k, getattr(self, k))
class switch(object):
    def __init__(self, value):
        self.value = value
        self.fall = False
    def __iter__(self):
        """Return the match method once, then stop"""
        yield self.match
        #raise StopIteration
    def match(self, *args):
        """Indicate whether or not to enter a case suite"""
        if self.fall or not args:
            return True
        elif self.value in args: # changed for v1.5, see below
            self.fall = True
            return True
        else:
            return False

opt = DefaultConfig()