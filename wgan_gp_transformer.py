from data_processing.data_preprocessing import preprocess_data_for_wgan_gp
from evaluation.cross_validation import perform_cross_validation
from models.wgan_gp import WGAN_GP
from models.wgan_gp import MLPBinaryClassifier
import torch
import pandas as pd
import numpy as np

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you have multi-GPU
np.random.seed(seed)

if __name__ == '__main__':
    data, labels, labels_for_wgan, scaler = preprocess_data_for_wgan_gp('merge.csv')
    data = data
    
    num_samples_list = [0, 100, 400, 700, 1000, 1300, 1600, 1900, 2200, 2500]
    
    model_params = {
        'input_dim': data.shape[1]-1,   #data.shape[1]-1
        'embed_dim':72,
        'nhead': 8, 
        'nhid': 16, 
        'nlayers': 6, 
        'dropout': 0.1,    
    }
    
    train_params = {
    'lr': 0.001,
    'epochs': 21,
    'patience': 2,
    }
    
    sample_filter_threshold = 0.0 #if you don't need a sample filter, set sample_filter_threshold=0.0
    
    results = perform_cross_validation(data=data,
                                       labels=labels,
                                       num_folds=5,
                                       num_samples_list=num_samples_list,
                                       model_params=model_params,
                                       train_params=train_params,
                                       scaler=scaler,
                                       train_batchsize=32,
                                       sample_filter_threshold=sample_filter_threshold) 
    #save the results to a csv file
    result = pd.DataFrame(results)
    if sample_filter_threshold == 0.0:
        result.to_csv('result_non_classif.csv',index=False,header=['num_samples', 'mean_accuracy', 'std_accuracy', 'mean_precision', 'std_precision', 'mean_recall', 'std_recall', 'mean_f1', 'std_f1'])
    else:
        result.to_csv('result.csv',index=False,header=['num_samples', 'mean_accuracy', 'std_accuracy', 'mean_precision', 'std_precision', 'mean_recall', 'std_recall', 'mean_f1', 'std_f1'])
    
                                          

