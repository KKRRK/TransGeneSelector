import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)

def delete_nonsence_data(data, column_name='label', random_state=42):
    threshold = 0.3
    mask = (
        (round(data[column_name]) == 0) & (abs(data[column_name] - 0) > threshold)
        | (round(data[column_name]) == 1) & (abs(data[column_name] - 1) > threshold)
        | (round(data[column_name]) <= -1)
        | (round(data[column_name]) > 1)
    )
    data_processed = data[~mask].copy() 
    data_processed['label'] = data_processed['label'].apply(lambda x: 1 if x >= 0.5 else 0)
    data_processed1 = data_processed[data_processed['label'] == 1]
    data_processed0 = data_processed[data_processed['label'] == 0]
    num = min(data_processed1.shape[0], data_processed0.shape[0])
    data_processed1 = data_processed1.sample(n=num, random_state=random_state)
    data_processed0 = data_processed0.sample(n=num, random_state=random_state)
    data_processed_left = pd.concat([data_processed1, data_processed0])
    print('delete nonsence data:', data_processed_left.shape[0])   
    return data_processed_left 


#随机抽样
def random_sample(data, n,random_state=42):
    data['label'] = data['label'].apply(lambda x: 1 if x >= 0.5 else 0)
    data_label1=data[data['label']==1]
    data_label0=data[data['label']==0]
    data_label1=data_label1.sample(n=n//2, random_state=random_state)
    data_label0=data_label0.sample(n=n//2, random_state=random_state)
    data_all=pd.concat([data_label1,data_label0])
    return data_all    


class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.model(x)
    
    
class WGAN_GP:
    def __init__(self, latent_dim=100, data_dim=200000, epochs=4000, lr=0.001):
        self.latent_dim = latent_dim
        self.data_dim = data_dim
        self.epochs = epochs
        self.lr = lr
        self.generator = Generator(latent_dim, data_dim)
        self.discriminator = Discriminator(data_dim)
        self.gen_optimizer = optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        self.disc_optimizer = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        print('Using device:', self.device)

    def train(self, train_loader):
        epoch_count, disc_loss_value, gen_loss_value = self.wgan_gp_train(train_loader, self.epochs, self.gen_optimizer, self.disc_optimizer, self.device)
        return epoch_count, disc_loss_value, gen_loss_value

    def wgan_gp_train(self, train_loader, epochs, gen_optimizer, disc_optimizer, device=None, gp_lambda=10):
        if device == None:
            device = self.device 
        self.generator = self.generator.to(device)
        self.discriminator = self.discriminator.to(device)
        epoch_count = []
        disc_loss_value = []
        gen_loss_value = []
        for epoch in range(epochs):
            for i, (real_data, _) in enumerate(train_loader):
                real_data = real_data.to(device)

                # train discriminator
                disc_optimizer.zero_grad()
                noise = torch.randn(real_data.shape[0], self.latent_dim, device=device)
                fake_data = self.generator(noise)

                real_output = self.discriminator(real_data)
                fake_output = self.discriminator(fake_data.detach())

                epsilon = torch.rand(real_data.size(0), 1, device=device).expand_as(real_data)  
                interp_data = epsilon * real_data.data + (1 - epsilon) * fake_data.data
                interp_data.requires_grad = True
                interp_output = self.discriminator(interp_data)
                gradients = torch.autograd.grad(outputs=interp_output, inputs=interp_data,
                                                grad_outputs=torch.ones_like(interp_output),
                                                create_graph=True, retain_graph=True, only_inputs=True)[0]
                
                gradient_penalty = gp_lambda * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
                disc_loss = fake_output.mean() - real_output.mean() + gradient_penalty
                disc_loss.backward()
                disc_optimizer.step()

                # train generator
                if i % 5 == 0:
                    gen_optimizer.zero_grad()
                    noise = torch.randn(real_data.shape[0], self.latent_dim, device=device)
                    fake_data = self.generator(noise)
                    fake_output = self.discriminator(fake_data)
                    gen_loss = -fake_output.mean()
                    gen_loss.backward()
                    gen_optimizer.step()
                    disc_loss_value.append(disc_loss.item())
                    gen_loss_value.append(gen_loss.item())
                    epoch_count.append(epoch)
                    print(f"Epoch: {epoch + 1}/{epochs}, Discriminator Loss: {disc_loss.item()}, Generator Loss: {gen_loss.item()}")
        return epoch_count, disc_loss_value, gen_loss_value
    


def generate_samples(generator,data, scaler,binary_classifier, selected_samples=5000, num_new_samples=500,latent_dim=100, sample_filter_threshold=0.1):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    filtered_generated_samples = pd.DataFrame() 
    filtered_generated_samples.rename(columns={0: 'label'}, inplace=True)
    generated_count = 0
    while True:
        # generate some new samples
        new_generated_samples = generator(torch.normal(0, 1, size=(num_new_samples, latent_dim)).to(device)).cpu().detach().numpy()
        # using the binary classifier to predict the labels of the new generated samples
        new_generated_samples_tensor = torch.from_numpy(new_generated_samples).float()
        with torch.no_grad():
            new_generated_predictions = binary_classifier(new_generated_samples_tensor.to(device))
            new_generated_labels = (new_generated_predictions >= sample_filter_threshold).cpu().float().numpy().squeeze() #.squeeze()函数将tensor的维度为1的维度去掉
        # select the samples that the classifier thinks are real
        filtered_generated_sample = new_generated_samples[new_generated_labels == 1,:].reshape(-1,data.shape[1])
        
        print(filtered_generated_sample.shape)

        generated_count += filtered_generated_sample.shape[0]

        # inverse z-score normalization
        inverse_scaled_tpm = scaler.inverse_transform(filtered_generated_sample)
        # inverse log transformation
        inverse_log_tpm = np.expm1(inverse_scaled_tpm)
        df_data = pd.DataFrame(inverse_log_tpm)
        df_data.rename(columns={df_data.columns[0]:'label'},inplace=True)
        df_data_left = delete_nonsence_data(df_data)
        filtered_generated_samples = pd.concat([filtered_generated_samples,df_data_left],axis=0)
        print('generated',len(filtered_generated_samples),'samples')
        try:
            df_data_random = random_sample(filtered_generated_samples,n=selected_samples)
            df_data_random = df_data_random.values 
            break
        except Exception as e:
            print(e)
            print('continue to generate samples')
            continue
     
    return df_data_random


class MLPBinaryClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
    

def train_binary_classifier(binary_classifier,generator,train_data,test_data,epochs=50,lr=0.001,latent_dim=100):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print('Using device:',device)
    # Train a binary classifier:
    # Hyperparameters
    epochs = epochs
    lr = lr
    num_generated_samples = len(train_data)
    X_train = train_data
    X_test = test_data
    
    # Initialize model and optimizer
    binary_classifier = binary_classifier  #MLPBinaryClassifier is a binary classifier, data_dim is the input dimension
    binary_classifier_optimizer = optim.Adam(binary_classifier.parameters(), lr=lr)
    # Prepare training and testing data
    generated_samples = generator(torch.randn(num_generated_samples, latent_dim).to(device)).cpu().detach().numpy()
    generated_labels = np.zeros(num_generated_samples)
    X_train_combined = np.vstack((X_train, generated_samples))
    y_train_combined = np.hstack((np.ones(len(X_train)), generated_labels)).reshape(-1, 1)
    X_train_combined_tensor = torch.from_numpy(X_train_combined).float().to(device)
    y_train_combined_tensor = torch.from_numpy(y_train_combined).float().to(device)
    print(y_train_combined_tensor.shape)
    X_test_gen= generator(torch.randn(len(X_test), latent_dim).to(device)).cpu().detach().numpy()
    y_test_gen = np.zeros(len(X_test))
 

    # Combine positive samples (real transcriptome data) with negative samples (generated data) to create a new training set:

    X_test_combined = np.vstack((X_test, X_test_gen))
    y_test_combined = np.hstack((np.ones(len(X_test)), y_test_gen)).reshape(-1, 1)

    X_test_tensor = torch.from_numpy(X_test_combined).float().to(device)
    y_test_tensor = torch.from_numpy(y_test_combined).float().to(device)

    # Define loss function
    criterion = nn.BCELoss() # Binary cross-entropy loss function

    # Train binary classifier
    for epoch in range(epochs):
        binary_classifier_optimizer.zero_grad()
        predictions = binary_classifier(X_train_combined_tensor)
        print(predictions.shape)
        
        loss = criterion(predictions, y_train_combined_tensor)
        loss.backward()
        binary_classifier_optimizer.step()

        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                test_predictions = binary_classifier(X_test_tensor)
                test_loss = criterion(test_predictions, y_test_tensor)
                accuracy = ((test_predictions > 0.5).float() == y_test_tensor).float().mean()
            print(f"Epoch: {epoch+1}/{epochs}, Train Loss: {loss.item()}, Test Loss: {test_loss.item()}, Test Accuracy: {accuracy.item()}")  