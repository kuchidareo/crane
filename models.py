import torch
import torch.nn as nn
import torch.nn.functional as F

class Base_CNN(nn.Module):
    def __init__(self, num_classes, num_columns, window_size=1, num_cnn_units=32, num_fc_units=128, dropout_rate=0.2):
        super(Base_CNN, self).__init__()

        self.num_classes = num_classes
        input_size = window_size * num_columns

        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=num_cnn_units, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=num_cnn_units, out_channels=num_cnn_units//2, kernel_size=3, stride=1, padding=1)
        
        output_size = self._get_conv_output_size(input_size)

        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(output_size, num_fc_units)
        self.fc2 = nn.Linear(num_fc_units, num_fc_units//2)
        self.fc3 = nn.Linear(num_fc_units//2, self.num_classes) 

    def _get_conv_output_size(self, input_size):
        # Create a dummy input tensor with the given window size
        x = torch.zeros(1, 1, input_size)
        
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        
        return x.size(1)
        
    def forward(self, x):
        x = x.view(x.size(0), 1, -1)  # Reshape the input tensor: (batch_size, 1, window*columns)
        
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        
        x = x.view(x.size(0), -1)
        
        x = self.dropout(self.elu(self.fc1(x)))
        x = self.dropout(self.elu(self.fc2(x)))
        x = self.fc3(x)
        return x

class LL_Arm_CNN(Base_CNN):
    def __init__(self, window_size=1, num_cnn_units=32, num_fc_units=128, dropout_rate=0.2):
        num_classes = 14
        num_columns = 38
        super(LL_Arm_CNN, self).__init__(num_classes, num_columns, window_size, num_cnn_units, num_fc_units, dropout_rate)
    
class Locomotion_CNN(Base_CNN):
    def __init__(self, window_size=1, num_cnn_units=32, num_fc_units=128, dropout_rate=0.2):
        num_classes = 5
        num_columns = 38
        super(Locomotion_CNN, self).__init__(num_classes, num_columns, window_size, num_cnn_units, num_fc_units, dropout_rate)

class Both_Arms_CNN(Base_CNN):
    def __init__(self, window_size=1, num_cnn_units=32, num_fc_units=128, dropout_rate=0.2):
        num_classes = 18
        num_columns = 160
        super(Both_Arms_CNN, self).__init__(num_classes, num_columns, window_size, num_cnn_units, num_fc_units, dropout_rate)

class Objects_CNN(Base_CNN):
    def __init__(self, window_size=1, num_cnn_units=32, num_fc_units=128, dropout_rate=0.2):
        num_classes = 24
        num_columns = 122
        super(Objects_CNN, self).__init__(num_classes, num_columns, window_size, num_cnn_units, num_fc_units, dropout_rate)

class Base_LSTM(nn.Module):
    def __init__(self, num_classes):
        super(Base_LSTM, self).__init__()

        self.num_classes = num_classes

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

class LL_Arm_LSTM(Base_LSTM):
    def __init__(self, num_classes=14):
        super(LL_Arm_LSTM, self).__init__(num_classes=num_classes)
        
class Locomotion_LSTM(Base_LSTM):
    def __init__(self, num_classes=5):
        super(LL_Arm_LSTM, self).__init__(num_classes=num_classes)