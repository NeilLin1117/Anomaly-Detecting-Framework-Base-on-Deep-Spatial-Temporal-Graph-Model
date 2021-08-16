## Anomaly-Detecting-Framework-Base-on-Deep-Spatial-Temporal-Graph-Model

## Introduction
An Anomaly Detecting Framework Base on DeepSpatial-Temporal Graph Model to detect malfunctioned air quality sensors.
[Paper Links](https://drive.google.com/file/d/19bCz4iNNMLlOtL3Ro27JT555vrMS_Eew/view)

## Quickstart
### GNN_reg<span></span>.py
* Training method is many to one (model separated).

Usage:
```
$python GNN_reg.py regression 
                --name <name of model, You can choose any name> 
                --model <model type, "gwnet" or "STGCN"> 
                --max_epoch <number of epoch>
                --use_gpu <weather use gpu>
                --num_nodes <number of graph node>
                --visual= <show figures during the training process, TRUE or FALSE>
```
Example:
```
$python GNN_reg.py regression 
                --name="GraphWaveNet_v1"
                --model="gwnet"
                --max_epoch=10
                --use_gpu=True
                --num_nodes=6
                --visual=False
```

### Global_GNN_reg<span></span>.py
* Training method is all to all.

Usage:
```
$python Global_GNN_reg.py regression 
                --name <name of model, You can choose any name> 
                --model <model type, "gwnet" or "STGCN"> 
                --max_epoch <number of epoch>
                --use_gpu <weather use gpu>
                --num_nodes <number of graph node>
```
Example:
```
$python Global_GNN_reg.py regression 
                --name="Global_GraphWaveNet_v1"
                --model="gwnet"
                --max_epoch=10
                --use_gpu=True
                --num_nodes=144
```

### Dep_GNN_reg<span></span>.py
* Training method is many to one (model uniform).

Usage:
```
$python Dep_GNN_reg.py regression 
                --name <name of model, You can choose any name> 
                --model <model type, "gwnet" or "STGCN"> 
                --max_epoch <number of epoch>
                --use_gpu <weather use gpu>
                --num_nodes <number of graph node>
```
Example:
```
$python Dep_GNN_reg.py regression 
                --name="Dep_GraphWaveNet_v1"
                --model="gwnet"
                --max_epoch=10
                --use_gpu=True
                --num_nodes=6
```
### deep_learning_reg<span></span>.py
* Training method is many to one (model separated).

Usage:
```
$python deep_learning_reg.py regression 
                --name <name of model, You can choose any name> 
                --model <model type, "TCN" or "LSTM" or "DNN"> 
                --max_epoch <number of epoch>
                --use_gpu <weather use gpu>
```
Example:
```
$python deep_learning_reg.py regression 
                --name="TCN_v1"
                --model="TCN"
                --max_epoch=10
                --use_gpu=True
```
### machine_learning_reg<span></span>.py
* Training method is many to one (model separated).

Usage:
```
$python machine_learning_reg.py regression 
                --name <name of model, You can choose any name> 
                --model <model type, "Lasso" or "Ridge" or "RandomForest"> 
```
Example:
```
$python machine_learning_reg.py regression 
                --name="Lasso_v1"
                --model="Lasso"
```
### Model parameters/ [description](https://hackmd.io/8tQ4zjZ-TG-bFzA3Uchumw?view) 
### Other hyperparameter setting please see the config.py file
## Datasets
### device_ground_truth.csv
csv format:
```
last_three_number, time, bias, device_ID
```
- last_three_number: The last three digits of the air quality sensor's type
- time: Inspection date of the Air Quality Sensors
- bias: Whether there is an abnormality, 1 means abnormal, 0 means normal
- device_ID: ID number of the Air Quality Sensors

### temporal_spatio_pm_2_5_144.gz
* a gz file store temporal spatio pm2.5 series datas of 144 devices. 

data format:
```
time , ID_1, ID_2, ID_3, ... , ID_144
00:00, 18.0, 17.0, 15.0, ... , 27.0
00:01, 17.0, 18.0, 10.0, ... , 33.0
00:02, 19.0, 17.0, 16.0, ... , 19.0
               .
               .
```
### normalized_laplacian_144.npy
* a npy file store a normalized laplacian graph matrix with 144 nodes.

### temporal_spatio_pm_2_5 
* a folder contain 144 csv files.
* each csv file contain temporal spatio pm2.5 series datas of 6 devices.
* label column mean pm2.5 series datas of the center device
* ID_1, ID_2, ... , ID_5 means the five nearest devices around center device

each csv format:
```
time , label, ID_1, ID_2, ... , ID_5
00:00, 18.0,  17.0, 15.0, ... , 27.0
00:01, 17.0,  18.0, 10.0, ... , 33.0
00:02, 19.0,  17.0, 16.0, ... , 19.0
               .
               .
```

### normalized_laplacian
* a folder contain 144 npy files.
* each npy file contain store a normalized laplacian graph matrix with 6 nodes.
* each npy file is the graph structure correspond to the csv file in temporal_spatio_pm_2_5 folder

## Project experiment environment  
- OS：  
    - Distributor ID: Ubuntu  
    - Description:    Ubuntu 18.04.4 LTS  
    - Release:        18.04  
    - Codename:       bionic  
- Python 3.6.9  
    - numpy: 1.17.0
    - pandas: 1.1.5
    - torch: 1.8.1
    - sklearn: 0.0
    - tqdm: 4.60.0
    - matplotlib: 3.1.2
    - fire: 0.4.0
    - joblib: 1.0.1
# Reference
1. https://github.com/chenyuntc/pytorch-book
2. https://github.com/nnzhan/Graph-WaveNet
3. https://github.com/Aguin/STGCN-PyTorch
4. https://github.com/locuslab/TCN
