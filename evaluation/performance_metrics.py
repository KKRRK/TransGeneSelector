import torch
import numpy as np
import pandas as pd
from scipy.linalg import sqrtm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import os
from evaluation.memory_record import print_memory_usage
from models.transformer_classif import TransformerModel,train_transformer
from data_processing.data_loader import CustomDataset


seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
np.random.seed(seed)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
def transpose_data(data):
    return data.transpose(1, 2)

def evaluate_model(train_data, train_labels,val_data, val_labels, num_samples, fold, model_params=None, train_params=None, train_batchsize=32,val_batchsize=32):
    # set default model_params
    if model_params is None:
        model_params = {
            'input_dim': data.shape[1],
            'embed_dim':500,
            'nhead': 2, 
            'nhid': 5, 
            'nlayers': 2, 
            'dropout': 0.1,    
        }
    
    model = TransformerModel(**model_params).to(device)
    print("evaluating model")
    print_memory_usage()
    
    # set default train_params
    if train_params is None:
        train_params = {
            'lr': 0.001,
            'epochs': 10,
        }
        
    train_data = train_data[:,1:]
    val_data = val_data[:,1:]
    
    print("Length of train_data:", len(train_data))
    print("Length of train_labels:", len(train_labels))
    train_dataset=CustomDataset(train_data, train_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=train_batchsize, shuffle=True)
    
    print("Number of batches in train_loader:", len(train_loader))
    

    val_dataset = CustomDataset(val_data, val_labels)
    val_loader = DataLoader(val_dataset, batch_size=val_batchsize)
    
    # train the model
    train_transformer(train_loader=train_loader, val_loader=val_loader, model=model, **train_params)

    model.eval()
    val_predictions = []
    for data, _ in val_loader:
        
        data = data.to(device)
        with torch.no_grad():
            outputs = model(data)
            probabilities = torch.sigmoid(outputs)  
            predicted = (probabilities > 0.5).float().squeeze() 
            val_predictions.extend(predicted.cpu().numpy())

    # cacuulate performance metrics
    accuracy = accuracy_score(val_labels, val_predictions)
    precision = precision_score(val_labels, val_predictions)
    recall = recall_score(val_labels, val_predictions)
    f1 = f1_score(val_labels, val_predictions)

    print("finished evaluating model...")
    print_memory_usage()
    
    
    if  os.path.exists('models/transformer_fold'+str(fold)+'+'+str(num_samples)+'.pth') == False:
        #保存模型
        torch.save(model.state_dict(), 'models/transformer_fold'+str(fold)+'+'+str(num_samples)+'.pth')
    
    return accuracy, precision, recall, f1



def UMAP_precess(real_data,generated_data):
    umap_model = umap.UMAP(n_components=2, random_state=42)
    real_data_umap = umap_model.fit_transform(real_data.astype(np.float64))
    generated_data_umap = umap_model.transform(generated_data.astype(np.float64))
    real_labels = real_data[:,0]
    generated_labels = generated_data[:,0]
    return real_data_umap,generated_data_umap,real_labels,generated_labels
    


def draw_UMAP(real_data_umap, generated_data_umap, real_labels, generated_labels):
    
    real_df = pd.DataFrame(real_data_umap, columns=['UMAP1', 'UMAP2'])
    real_df['label'] = real_labels
    real_df['type'] = 'Real'

    generated_df = pd.DataFrame(generated_data_umap, columns=['UMAP1', 'UMAP2'])
    generated_df['label'] = generated_labels
    generated_df['type'] = 'Generated'

    combined_df = pd.concat([real_df, generated_df], ignore_index=True) 
    combined_df.to_csv('combined_df.csv',index=False)
  
  
    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=combined_df, x='UMAP1', y='UMAP2', hue='label', style='type', palette='Set1',alpha=0.8,s=30)
    plt.title('UMAP Projection of Real and Generated Data')
    plt.show()

