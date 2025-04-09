from torch.utils.data import Dataset, DataLoader
import torch
from utils import preprocess_audio

class DummyDataset(Dataset):
    def __init__(self, num_samples=1000, input_size=128):
        self.num_samples = num_samples
        self.input_size = input_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate random data
        x = torch.randn(self.input_size)
        y = torch.randn(self.input_size)

        x = x.unsqueeze(0)  # Add batch dimension
        # y = y.unsqueeze(0)
        return x, y
    
def get_dummy_dataloaders(batch_size=32, num_samples=1000, input_size=128):
    dataset = DummyDataset(num_samples=num_samples, input_size=input_size)
    test_dataset = DummyDataset(num_samples=num_samples, input_size=input_size)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader
    