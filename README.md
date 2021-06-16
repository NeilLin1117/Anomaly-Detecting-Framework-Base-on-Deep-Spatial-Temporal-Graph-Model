## Anomaly-Detecting-Framework-Base-on-Deep-Spatial-Temporal-Graph-Model

## Introduction
Detecting Malfunctioned Air Quality Sensors – anAnomaly Detecting Framework Base on DeepSpatial-Temporal Graph Model.

## Quickstart
### GNN_reg<span></span>.py

### Global_GNN_reg<span></span>.py

### Dep_GNN_reg<span></span>.py

### deep_learning_reg<span></span>.py

### machine_learning_reg<span></span>.py

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
* a npy file store a normalized laplacian graph structure with 144 nodes.

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
* each npy file contain store a normalized laplacian graph structure with 6 nodes.
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
