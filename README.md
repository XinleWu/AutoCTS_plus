# Joint Neural Architecture and Hyperparameter Search for Correlated Time Series Forecasting

This is the original pytorch implementation of SEARCH in the following paper: Joint Neural Architecture and Hyperparameter Search for Correlated Time Series Forecasting.

## Requirements
- python 3.6
- see `requirements.txt`
## Data Preparation
SEARCH is implemented on several public correlated time series forecasting datasets.

- **PEMS03**, **PEMS04**, **PEMS07** and **PEMS08** from [STSGCN (AAAI-20)](https://github.com/Davidham3/STSGCN).
Download the data [STSGCN_data.tar.gz](https://pan.baidu.com/s/1ZPIiOM__r1TRlmY4YGlolw) with password: `p72z` and uncompress data file using`tar -zxvf data.tar.gz`

- **Solar-Energy** and **Electricity** datasets from [https://github.com/laiguokun/multivariate-time-series-data](https://github.com/laiguokun/multivariate-time-series-data). Uncompress them and move them to the data folder.


## Architecture Search
#### search for the best arch-hyper on the PEMS08 dataset
```
mkdir saved_model
CUDA_VISIBLE_DEVICES=0 python3.6 joint_search.py
```
## Architecture test
#### test the arch-hyper searched on the PEMS08 dataset
```
cd test
CUDA_VISIBLE_DEVICES=0 python3.6 joint_test.py
```

# Reference

If you use AutoCTS+ for your research, please cite the following paper. 
<pre>     
@article{wu2023autocts+,
  title={AutoCTS+: Joint neural architecture and hyperparameter search for correlated time series forecasting},
  author={Wu, Xinle and Zhang, Dalin and Zhang, Miao and Guo, Chenjuan and Yang, Bin and Jensen, Christian S},
  journal={Proceedings of the ACM on Management of Data},
  volume={1},
  number={1},
  pages={1--26},
  year={2023},
  publisher={ACM New York, NY, USA}
}
</pre>   
