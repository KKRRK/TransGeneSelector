from torch.utils.data import Dataset
import torch

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data = torch.tensor(data, dtype=torch.float32).to(device)
        self.labels = torch.tensor(labels, dtype=torch.long).to(device)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    

