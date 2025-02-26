import torch.nn as nn
import torch.optim as optim
import torch
import math
import numpy as np


seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
        
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
        
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x
    
class TransformerModel(nn.Module):
    def __init__(self, input_dim, embed_dim, nhead, nhid, nlayers, batch_first=True, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.batch_first = batch_first
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.activation = nn.ReLU()  
        self.pos_encoder = PositionalEncoding(embed_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=nhid, dropout=dropout),
            num_layers=nlayers
        )
        self.classifier = nn.Linear(embed_dim, 1)
        self.embedding.apply(init_weights)
        self.transformer.apply(init_weights)
        self.classifier.apply(init_weights)

    def forward(self, x):
        x = x.unsqueeze(2)  
        if self.batch_first:
            x = x.transpose(1, 2)  

        x = self.embedding(x)
        x = self.activation(x)  # Apply activation function before positional encoding
        if self.batch_first:
            x = x.transpose(1, 2)  # Transpose dimensions
            # print (x.shape)
        x = self.pos_encoder(x)
        
        if self.batch_first:
            x = x.transpose(0, 1)  # Transpose dimensions (batch_size, seq_len, embed_dim) -> (seq_len, batch_size, embed_dim)
            # print (x.shape)
        x = self.transformer(x)
        
        if self.batch_first:
            x = x.transpose(0, 1)  # Transpose dimensions (seq_len, batch_size, embed_dim) -> (batch_size, seq_len, embed_dim)
            # print ('end',x.shape)
        x = self.classifier(x[:, 0])
        return x
 
def transpose_data(data):
    return data.transpose(1, 2)   
    
def train_transformer(model, train_loader, val_loader, epochs, lr, patience):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print('Using device:', device)
    criterion = nn.BCEWithLogitsLoss()  
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    patience_counter = 0
    for epoch in range(epochs):
        model.train()
        for i, (data, labels) in enumerate(train_loader):
            data, labels = data.to(device), labels.to(device)
            
            print(data.shape)

            optimizer.zero_grad()
            outputs = model(data)
            outputs = outputs.squeeze()  
            loss_train = criterion(outputs, labels.float()) 
            loss_train.backward()
            optimizer.step()

            print(f'Epoch [{epoch}/{epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss_train.item()}')

        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, labels in val_loader:
                data, labels = data.to(device), labels.to(device)

                outputs = model(data)
                outputs = outputs.squeeze()  
                loss = criterion(outputs, labels.float())  
                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f'Epoch [{epoch}/{epochs}], Validation Loss: {val_loss}')
        

        # early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        if val_loss >= best_val_loss:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping due to no improvement in validation loss")
            break

