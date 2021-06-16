## Anomaly-Detecting-Framework-Base-on-Deep-Spatial-Temporal-Graph-Model

## Introduction
Detecting Malfunctioned Air Quality Sensors – anAnomaly Detecting Framework Base on DeepSpatial-Temporal Graph Model.

## Datasets
#### device_ground_truth.csv
csv format:
```
last_three_number, time, bias, device_ID
```
- last_three_number: The last three digits of the air quality sensor's type
- time: Inspection date of the Air Quality Sensors
- bias: Whether there is an abnormality, 1 means abnormal, 0 means normal
- device_ID: ID number of the Air Quality Sensors

#### normalized_laplacian_144.npy
* a npy file store a normalized laplacian graph structure with 144 nodes

#### normalized_laplacian_144.npy
* a npy file store a normalized laplacian graph structure with 144 nodes

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
