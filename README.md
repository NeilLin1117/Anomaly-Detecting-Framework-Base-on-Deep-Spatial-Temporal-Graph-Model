## Anomaly-Detecting-Framework-Base-on-Deep-Spatial-Temporal-Graph-Model

## Introduction
Detecting Malfunctioned Air Quality Sensors – anAnomaly Detecting Framework Base on DeepSpatial-Temporal Graph Model.

## Datasets
#### device_ground_truth.csv/
Data format:
```
last_three_number time bias device_ID
```
- last_three_number: 空汙感測器其型號末三碼
- time: 空汙感測器巡檢日期
- bias: 是否出現異常，1代表異常，0代表正常
- device_ID: 空汙感測器ID編號

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
