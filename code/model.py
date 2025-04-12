import torch
import torch.nn as nn


class LSTMFilterNetwork(nn.Module):
    def __init__(self, input_size=128, hidden_size=200, output_size=128):
        super(LSTMFilterNetwork, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(output_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, hidden=None):
        if hidden is None:
            lstm_out, hidden = self.lstm(x)
        else:
            lstm_out, hidden = self.lstm(x, hidden)
        x = self.fc1(lstm_out[:, -1, :])
        x = self.relu(x)
        x = self.fc2(x)
        return self.sigmoid(x), hidden
    

def load_model(model_path=None):
    if model_path is None:
        return LSTMFilterNetwork()
    
    # Load the model state dictionary
    model = LSTMFilterNetwork()
    model.load_state_dict(torch.load(model_path))
    return model