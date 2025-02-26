import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from evaluation.memory_record import print_memory_usage


seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)


def preprocess_data_for_wgan_gp(filepath):
    print("start preprocessing data...") #start preprocessing data
    print_memory_usage()
    data = pd.read_csv(filepath, index_col=None, header=None)

    data = data.T
    data.columns = data.iloc[0]
    data = data.drop(data.index[0])

    data_copy = data.copy()
    data_copy = data_copy.rename(columns={data_copy.columns[0]: 'label'})

    data = data.apply(pd.to_numeric, errors='coerce')
    data = data.astype(np.float32)

    data = data.to_numpy()
    data = np.log1p(data)  # log1p(x) = log(1 + x)
    
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    print('!!!!!!!!!!!',data.shape)
    
    labels = data_copy.iloc[:, 0].astype(int).tolist() 

    labels_for_wgan = np.ones((data.shape[0], 1))
    
    print("finished preprocessing data")
    print_memory_usage()
    return data, labels, labels_for_wgan, scaler


def preprocess_data_for_test(test_filepath, scaler):
    print("start preprocessing data...") 
    print_memory_usage()

    test_data = pd.read_csv(test_filepath, index_col=None, header=None)
    test_data = test_data.T
    test_data.columns = test_data.iloc[0]
    test_data = test_data.drop(test_data.index[0])
    test_data_copy = test_data.copy()
    test_data = test_data.apply(pd.to_numeric, errors='coerce')
    test_data = test_data.astype(np.float32)
    test_data = test_data.to_numpy()
    test_data = np.log1p(test_data)
    test_data = scaler.transform(test_data)

    print('!!!!!!!!!!!',test_data.shape)
    
    test_label = test_data_copy.iloc[:, 0].astype(int).tolist() 

    print("finished preprocessing data")
    print_memory_usage()
    return test_data, test_label



def delete_nonsence_data(data, column_name='label',threshold=0.2):
    mask = (
        (round(data[column_name]) == 0) & (abs(data[column_name] - 0) > threshold)
        | (round(data[column_name]) == 1) & (abs(data[column_name] - 1) > threshold)
        | (round(data[column_name]) <= -1)
        | (round(data[column_name]) > 1)
    )
    data_processed = data[~mask].copy() 
    print('删除异常值后的数据量：', data_processed.shape[0])
    return data_processed