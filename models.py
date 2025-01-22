import torch
import torch.nn as nn
import torch.nn.functional as F

class LL_Arm_CNN(nn.Module):
    def __init__(self, window_size=1):
        super(LL_Arm_CNN, self).__init__()

        self.num_classes = 14 # LL_ARM has 13 classes
        
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)  # Max pooling
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        
        self.window_size = window_size
        self.output_size = self._get_conv_output_size(window_size)

        self.fc1 = nn.Linear(self.output_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, self.num_classes) 

    def _get_conv_output_size(self, window_size):
        # Create a dummy input tensor with the given window size
        x = torch.zeros(1, 1, window_size*38)
        
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        
        return x.size(1)
        
    def forward(self, x):
        x = x.view(x.size(0), 1, -1)  # Reshape the input tensor: (batch_size, 1, window*columns)
        
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class Locomotion_CNN(nn.Module):
    def __init__(self, window_size=1):
        super(Locomotion_CNN, self).__init__()

        self.num_classes = 5 # Locomotion has 5 classes
        
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)  # Max pooling
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)

        self.window_size = window_size
        self.output_size = self._get_conv_output_size(window_size)
        
        self.fc1 = nn.Linear(self.output_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, self.num_classes)

    def _get_conv_output_size(self, window_size):
        # Create a dummy input tensor with the given window size
        x = torch.zeros(1, 1, window_size*38)
        
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        
        return x.size(1)
        
    def forward(self, x):
        x = x.view(x.size(0), 1, -1)  # Reshape the input tensor: (batch_size, 1, window*columns)
        
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class LL_Arm_LSTM(nn.Module):
    def __init__(self):
        super(LL_Arm_LSTM, self).__init__()

        self.num_classes = 14 

        self.lstm1 = nn.LSTM(input_size=38, hidden_size=50, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(input_size=50, hidden_size=15, batch_first=True)
        self.fc = nn.Linear(15, self.num_classes)  # Output layer

    def forward(self, x):
        # x.size(): (batch, window, columns) [32, 32, 38]
        out, _ = self.lstm1(x)
        out = self.dropout(out)
        out, _ = self.lstm2(out)
        out = self.dropout(out)
        out = self.fc(out[:, -1, :])  # Get the last time step output
        return out
        
class Locomotion_LSTM(nn.Module):
    def __init__(self):
        super(Locomotion_LSTM, self).__init__()

        self.num_classes = 5
        
        self.lstm1 = nn.LSTM(input_size=38, hidden_size=50, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(input_size=50, hidden_size=50, batch_first=True)
        self.fc = nn.Linear(50, self.num_classes)  # Output layer

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout(out)
        out, _ = self.lstm2(out)
        out = self.dropout(out)
        out = self.fc(out[:, -1, :])  # Get the last time step output
        return out