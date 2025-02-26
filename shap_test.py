import numpy as np
import shap
import pandas as pd
import torch
import sys
import os
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import seaborn as sns
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
from models.transformer_classif import TransformerModel

def preprocess_data_for_wgan_gp(filepath):
    print("Start preprocessing data...")
    data = pd.read_csv(filepath, index_col=None, header=None)
    # Transpose
    data = data.T
    data.columns = data.iloc[0]
    data = data.drop(data.index[0])

    data_copy = data.copy()
    data_copy = data_copy.rename(columns={data_copy.columns[0]: 'label'})

    # Drop non-numeric columns (e.g., if the first column is a string type label)
    # data = data.iloc[:, 1:]
    data = data.apply(pd.to_numeric, errors='coerce')
    data = data.astype(np.float32)

    # Convert to NumPy array
    data = data.to_numpy()

    # Standardize data, first transform data with log, then standardize
    data = np.log1p(data)  # log1p(x) = log(1 + x)
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    print('!!!!!!!!!!!',data.shape)
    
    labels = data_copy.iloc[:, 0].astype(int).tolist()  #round() rounds to the nearest integer, astype(int) converts to integer
    gene_names = data_copy.columns[1:].tolist()
    # Create label array
    labels_for_wgan = np.ones((data.shape[0], 1))
    
    print("Data preprocessing completed")
    return data, labels, labels_for_wgan, scaler, gene_names

def get_shap_values(model, X):
    # Initialize an explainer with the SHAP library
    explainer = shap.explainers.Deep(model, X)
    # Compute SHAP values
    shap_values = explainer.shap_values(X)
    return shap_values,explainer

def get_important_genes(shap_values, gene_names, top_n=100):
    # Compute the mean absolute SHAP value for each gene
    mean_abs_shap_values = np.mean(np.abs(shap_values), axis=0)
    # Sort SHAP values and get the indices of the top n most important genes
    top_gene_indices = np.argsort(mean_abs_shap_values)[-top_n:][::-1]
    print(top_gene_indices)
    # Get the names of the most important genes
    important_genes = [gene_names[i] for i in top_gene_indices]
    importances = [mean_abs_shap_values[i] for i in top_gene_indices]
    return important_genes, importances

def plot_shap_values(shap_values, data, gene_names, top_n=30):
    # Get the indices of the most important genes
    mean_abs_shap_values = np.mean(np.abs(shap_values), axis=0)
    top_gene_indices = np.argsort(mean_abs_shap_values)[-top_n:][::-1]
    feature_names=np.array(gene_names)[top_gene_indices]
    data = np.array(data)[:,top_gene_indices]
    # Merge data and feature_names
    # Create a summary plot
    shap.summary_plot(shap_values[:, top_gene_indices],features=data, feature_names=feature_names, max_display=top_n)
    plt.show()

def plot_force_values(explainer, shap_values, gene_names, top_n=500):
    # Get the indices of the most important genes
    mean_abs_shap_values = np.mean(np.abs(shap_values), axis=0)
    top_gene_indices = np.argsort(mean_abs_shap_values)[-top_n:][::-1]

    # Select the SHAP values for one or more samples
    selected_shap_values = shap_values[0:10, top_gene_indices]  # Select the SHAP values for the first sample

    # Create a force plot
    force_plot = shap.force_plot(explainer.expected_value, selected_shap_values, feature_names=np.array(gene_names)[top_gene_indices])

    # Save as an HTML file
    shap.save_html("force_plot.html", force_plot)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Read data
    data,labels,_,_,gene_names = preprocess_data_for_wgan_gp("merge.csv")
    data = data[:,1:]
     # Convert to tensor
    data_copy = data.copy()
    data = torch.tensor(data, dtype=torch.float32).to(device)
    # Remove all data without labels
    # Print the first five rows of data
    print(data)
    # Load model
    model_params = {
    'input_dim': data.shape[1],   #data.shape[1]-1
    'embed_dim':72,
    'nhead': 8, 
    'nhid': 16, 
    'nlayers': 6, 
    'dropout': 0.1,    
    }
    
    transformer_model = TransformerModel(**model_params).to(device)
    
    # Load model
    model_dict = torch.load('models/transformer_fold4+2200.pth')   
    transformer_model.load_state_dict(model_dict)

    # Compute SHAP values
    shap_values,explainer = get_shap_values(transformer_model, data)

    # Get the most important genes
    top_n = 1000  # Number of important genes
    important_genes, importances = get_important_genes(shap_values, gene_names, top_n)
    # Visualize SHAP values
    # shap.summary_plot(shap_values,data)
    data = data.cpu()
    plot_shap_values(shap_values=shap_values, data=data, gene_names=gene_names, top_n=top_n)
    # plot_force_values(explainer,shap_values, gene_names, top_n=50)
    # Print the most important genes
    print("Important genes:")
    for i, gene in enumerate(important_genes, start=1):
        print(f"{i}. {gene}")
        
    # Save the important genes
    df = pd.DataFrame({'gene':important_genes,'importance':importances})
    df.to_csv('important_genes_by_shap.csv',index=False)
    
    # Plot the importance of the important genes
    sns.barplot(x=importances[:50],y=important_genes[:50])
    plt.show()
