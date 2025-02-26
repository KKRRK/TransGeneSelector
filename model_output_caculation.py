import torch
import os
import pandas as pd
from models.transformer_classif import TransformerModel
from data_processing.data_preprocessing import preprocess_data_for_wgan_gp
from data_processing.data_preprocessing import preprocess_data_for_test
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


data, labels, labels_for_wgan, scaler = preprocess_data_for_wgan_gp('merge.csv')
test_data, test_labels = preprocess_data_for_test('test_merge.csv', scaler=scaler)
test_data = test_data[:,1:]
print(test_data.shape)

model_params = {
'input_dim': data.shape[1]-1,   #data.shape[1]-1
'embed_dim':72,
'nhead': 8, 
'nhid': 16, 
'nlayers': 6, 
'dropout': 0.1,    
}

pth_files = [file for file in os.listdir('models') if file.endswith('.pth')]
data = {'labels':test_labels}
for file in pth_files:
    transformer_model = TransformerModel(**model_params).to(device)
    model_dict = torch.load('models/'+file)
    transformer_model.load_state_dict(model_dict)

    transformer_model.eval()
    transformer_model.to(device)

    test_data = torch.tensor(test_data, dtype=torch.float32).to(device)

    with torch.no_grad():
        outputs = transformer_model(test_data)
        probabilities = torch.sigmoid(outputs.data)
        probabilities = torch.squeeze(probabilities) 
        print(probabilities)
        data[file] = probabilities.cpu().tolist()
        print(data[file] )
        
data_copy = data.copy()
index = {'accuracy':[],'precision':[],'recall':[],'f1':[]}

#Caculate index
for key in data.keys():
    data[key] = [1 if i>0.5 else 0 for i in data[key]]
    print(key)
    print('accuracy:',accuracy_score(data['labels'], data[key]))
    index['accuracy'].append(accuracy_score(data['labels'], data[key]))
    print('precision:',precision_score(data['labels'], data[key]))
    index['precision'].append(precision_score(data['labels'], data[key]))
    print('recall:',recall_score(data['labels'], data[key]))
    index['recall'].append(recall_score(data['labels'], data[key]))
    print('f1:',f1_score(data['labels'], data[key]))
    index['f1'].append(f1_score(data['labels'], data[key]))
    
models = {'models':[key for key in data.keys()] }

#Save output result
models = pd.DataFrame(models)
index = pd.DataFrame(index)
data_index = pd.concat([models,index],axis=1)
    
data = pd.DataFrame(data_copy)
data.to_csv('output_result.csv',index=False)
data_index.to_csv('output_indexes.csv',index=False)
print(test_data.shape)
    
    

