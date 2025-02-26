import torch
import numpy as np
from models.wgan_gp import WGAN_GP
from models.wgan_gp import MLPBinaryClassifier
from data_processing.data_loader import CustomDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from evaluation.performance_metrics import UMAP_precess
from evaluation.performance_metrics import draw_UMAP
from models.wgan_gp import generate_samples
from data_processing.data_preprocessing import preprocess_data_for_wgan_gp

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

if __name__ == '__main__':
    data, labels, labels_for_wgan, scaler = preprocess_data_for_wgan_gp('merge.csv')
    data = data
    
    # Instantiate WGAN_GP
    wgan_gp = WGAN_GP(latent_dim=100,data_dim=data.shape[1],epochs=3800,lr=0.001)
    
    # Instantiate binary classifier
    binary_classifier = MLPBinaryClassifier(input_dim=data.shape[1]).to(device)
    
    # Split data into training and testing sets
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels_for_wgan, test_size=0.2, random_state=42)
    
    # Load data
    train_dataset = CustomDataset(train_data, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Train WGAN_GP
    wgan_gp.train(train_loader)
    
    # Generate samples
    generate_sample = generate_samples(generator=wgan_gp.generator, data=data, scaler=scaler,binary_classifier=binary_classifier, selected_samples=100, num_new_samples=500, latent_dim=100)
    generate_sample[:,0] = [1 if generate_sample[:,0][i] >= 0.5 else 0 for i in range(len(generate_sample))]
    
    # Inverse z-score normalization
    train_data = scaler.inverse_transform(train_data)
    # Inverse log transformation
    train_data = np.expm1(train_data)
    
    train_data[:,0] = [1 if train_data[:,0][i] >= 0.5 else 0 for i in range(len(train_data))]
    
    # UMAP dimensionality reduction
    real_data_umap,generated_data_umap,real_labels,generated_labels = UMAP_precess(train_data, generate_sample)
    
    draw_UMAP(real_data_umap,generated_data_umap,real_labels,generated_labels)
    