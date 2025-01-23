import warnings

from bayes_opt import BayesianOptimization
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

import value
from preprocess import DataProcessor
import models

warnings.filterwarnings('ignore')

position = "left_arm" # Change
window_size = 32
model_type = "CNN"


processor = DataProcessor()
train_df = pd.read_csv(f"{value.traindata_filename}.csv")
test_df = pd.read_csv(f"{value.testdata_filename}.csv")

sensors = processor.position_to_sensor[position]
columns = [processor.sensor_to_column[sensor] for sensor in sensors]
flatten_columns = [item for row in columns for item in row]
position_label_column = processor.position_to_original_position_label[position]

X = np.asarray(train_df[flatten_columns])
y = np.asarray(train_df[position_label_column])
X_test = np.asarray(test_df[flatten_columns])
y_test = np.asarray(test_df[position_label_column])

X, y = processor.transform_to_window_data(X, y, window_size=window_size)
X_test, y_test = processor.transform_to_window_data(X_test, y_test, window_size=window_size)

X_train, X_val, y_train, y_val = train_test_split(X, y, shuffle=False, test_size=0.20)

def create_dataloader(X, y, batch_size=32, shuffle=True):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

def evaluate_accuracy(model, dataloader):
    non_zero_loss = 0
    non_zero_correct = 0
    non_zero_total = 0
    overall_loss = 0
    overall_correct = 0
    overall_total = 0
    predicted_labels = []
    confusion_matrix = np.zeros((model.num_classes, model.num_classes))

    for X_batch, y_batch in tqdm(dataloader, desc='Validation'):
        with torch.no_grad():
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)

            predicted_labels.extend(predicted.tolist())

            # Calculate overall accuracy and loss
            overall_total += y_batch.size(0)
            overall_correct += (predicted == y_batch).sum().item()
            overall_loss += torch.nn.functional.cross_entropy(outputs, y_batch).item()

            # Filter out samples where the label is 0
            mask = (y_batch != 0)
            filtered_y_batch = y_batch[mask]
            filtered_predicted = predicted[mask]

            # Calculate loss and accuracy only for non-zero labels
            if len(filtered_y_batch) > 0:
                loss = torch.nn.functional.cross_entropy(outputs[mask], filtered_y_batch)
                non_zero_loss += loss.item()
                non_zero_total += filtered_y_batch.size(0)
                non_zero_correct += (filtered_predicted == filtered_y_batch).sum().item()

            # Update confusion matrix
            for i, j in zip(y_batch, predicted):
                confusion_matrix[i, j] += 1

    non_zero_val_loss = non_zero_loss / non_zero_total if non_zero_total > 0 else float('inf')
    non_zero_val_accuracy = non_zero_correct / non_zero_total if non_zero_total > 0 else 0.0
    overall_loss = overall_loss / overall_total if overall_total > 0 else float('inf')
    overall_accuracy = overall_correct / overall_total if overall_total > 0 else 0.0
    return non_zero_val_loss, non_zero_val_accuracy, overall_loss, overall_accuracy, predicted_labels, confusion_matrix

def train(model, dataloader, val_dataloader, epochs=10, patience=5, lr=0.001):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(epochs):
        for X_batch, y_batch in tqdm(dataloader, desc=f'Epoch {epoch + 1}/{epochs}'):
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = torch.nn.functional.cross_entropy(outputs, y_batch)
            loss.backward()
            optimizer.step()

        non_zero_val_loss, non_zero_val_accuracy, overall_val_loss, overall_val_accuracy, _, _ = evaluate_accuracy(model, val_dataloader)
        # Early stopping logic
        if overall_val_loss < best_val_loss:
            best_val_loss = overall_val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f'Early stopping triggered after {epoch + 1} epochs.')
                break

def test(model, dataloader):
    _, non_zero_val_accuracy, _, overall_val_acccuracy, _, _ = evaluate_accuracy(model, dataloader)
    return overall_val_acccuracy

def pipeline(num_cnn_units, num_fc_units, dropout_rate, batch_size, lr):
    num_cnn_units = int(num_cnn_units)
    num_fc_units = int(num_fc_units)
    batch_size = int(batch_size)

    train_dataloader = create_dataloader(X_train, y_train, batch_size=batch_size, shuffle=True)
    val_dataloader = create_dataloader(X_val, y_val, batch_size=batch_size, shuffle=False)
    test_dataloader = create_dataloader(X_test, y_test, batch_size=batch_size, shuffle=False)

    if position in [value.RIGHT_ARM, value.LEFT_ARM]:
        if model_type == 'CNN':
            model = models.LL_Arm_CNN(window_size=window_size, num_cnn_units=num_cnn_units, num_fc_units=num_fc_units, dropout_rate=dropout_rate)
        elif model_type == 'LSTM':
            model = models.LL_Arm_LSTM()
    else:
        if model_type == 'CNN':
            model = models.Locomotion_CNN(window_size=window_size, num_cnn_units=num_cnn_units, num_fc_units=num_fc_units, dropout_rate=dropout_rate)
        elif model_type == 'LSTM':
            model = models.Locomotion_LSTM()

    train(model, train_dataloader, val_dataloader, epochs=10, patience=5, lr=lr)
    test_overall_accuracy = test(model, test_dataloader)
    return test_overall_accuracy

def main():
    # Define parameter bounds for optimization
    pbounds = {
        'num_cnn_units': (256, 1024),
        'num_fc_units': (256, 1024),
        'dropout_rate': (0.1, 0.5),
        'batch_size': (32, 512),
        'lr': (1e-6, 1e-2),
    }

    # Create BayesianOptimization instance
    optimizer = BayesianOptimization(
        f=pipeline,
        pbounds=pbounds,
        verbose=2
    )
    optimizer.maximize(init_points=10, n_iter=10)
    print(optimizer.max)

if __name__ == "__main__":
    main()