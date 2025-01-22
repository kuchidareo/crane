import os
import warnings

import hydra
import mlflow
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

import value
from preprocess import DataProcessor
import models
from util import log_params_from_omegaconf_dict


warnings.filterwarnings('ignore')

def create_dataloader(X, y, batch_size=32, shuffle=True):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

def test(model, dataloader):
    _, non_zero_val_accuracy, _, overall_val_acccuracy, _, _ = evaluate_accuracy(model, dataloader)
    
    mlflow.log_metrics(
        {
            "non_zero_test_accuracy": non_zero_val_accuracy,
            "overall_test_accuracy": overall_val_acccuracy
        }
    )

    print(f'Non-Zero Test Acc. {non_zero_val_accuracy:.4f}, Overall Test Acc.: {overall_val_acccuracy:.4f}')

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

def train(model, dataloader, val_dataloader, epochs=10, patience=5):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
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
        mlflow.log_metrics(
            {
                "non_zero_val_loss": non_zero_val_loss,
                "non_zero_val_accuracy": non_zero_val_accuracy,
                "overall_accuracy": overall_val_accuracy,
                "overall_loss": overall_val_loss
            }, step=epoch
        )
        print(f'Epoch {epoch + 1}/{epochs}, Non-Zero Val Loss: {non_zero_val_loss:.4f}, Non-Zero Val Acc.: {non_zero_val_accuracy:.4f}, Overall Loss: {overall_val_loss:.4f}, Overall Acc.: {overall_val_accuracy:.4f}, ')

        # Early stopping logic
        if overall_val_loss < best_val_loss:
            best_val_loss = overall_val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f'Early stopping triggered after {epoch + 1} epochs.')
                break

def save_model(model, model_name):
    torch.save(model.state_dict(), f'trained_model/{model_name}')

def load_model(model, model_name):
    model.load_state_dict(torch.load(f'trained_model/{model_name}'))

def save_testset_prediction_accuracy(confusion_matrix, config):
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix, annot=True, fmt="g", cmap='Blues')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.savefig(f'fig/confusion_matrix_{config.model_type}_{config.window_size}_{config.position}.png')
    # plt.show()

def generate_testset_with_prediction_label(config, processor, model, test_dataloader):
    original_position = processor.position_to_original_position_label[config.position]
    
    _, non_zero_acc, _, acc, predicted_labels, confusion_matrix = evaluate_accuracy(model, test_dataloader)
    predicted_original_labels = [processor.reset_label_to_original_label[original_position][label] for label in predicted_labels]
    save_testset_prediction_accuracy(confusion_matrix, config)

    testset_with_prediction_filename = f'{value.testdata_with_prediction_filename}_{config.model_type}_{config.window_size}'
    file_path = testset_with_prediction_filename if os.path.exists(f"{testset_with_prediction_filename}.csv") else value.testdata_filename
    df = pd.read_csv(f"{file_path}.csv")

    df[original_position] = df[original_position].map(processor.reset_label_to_original_label[original_position])
    predicted_labels_column = [0 for _ in range(len(df))]
    for i in range(0, len(df) - config.window_size, config.window_size):
        if i < len(predicted_original_labels):
            predicted_labels_column[i:i+config.window_size] = [predicted_original_labels[i] for _ in range(config.window_size)]
        elif i == len(predicted_original_labels):
            predicted_labels_column = predicted_labels_column[:i]
    df[f'{original_position}_Predicted'] = predicted_labels_column

    if config.position in [value.RIGHT_ARM, value.LEFT_ARM]:
        df[original_position] = df[f'{original_position}'].astype(int)
        df[f'{original_position}_Predicted'] = df[f'{original_position}_Predicted'].astype(int)
    print(df[f'{original_position}'].value_counts())
    print(df[f'{original_position}_Predicted'].value_counts())
    df.to_csv(f"{testset_with_prediction_filename}.csv", index=False)

@hydra.main(config_name="default.yaml", config_path="conf", version_base=None)
def main(config):
    print(OmegaConf.to_yaml(config))
    mlflow.set_experiment(config.experiment_name)

    processor = DataProcessor()
    train_df = pd.read_csv(f"{value.traindata_filename}.csv")
    test_df = pd.read_csv(f"{value.testdata_filename}.csv")
   
    window_size = config.window_size
    model_type = config.model_type
    position = config.position
    model_name = f"{model_type}_{window_size}_{position}.pth"

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
    train_dataloader = create_dataloader(X_train, y_train, batch_size=32, shuffle=True)
    val_dataloader = create_dataloader(X_val, y_val, batch_size=32, shuffle=False)
    test_dataloader = create_dataloader(X_test, y_test, batch_size=32, shuffle=False)

    if position in [value.RIGHT_ARM, value.LEFT_ARM]:
        if model_type == 'CNN':
            model = models.LL_Arm_CNN(window_size=window_size)
        elif model_type == 'LSTM':
            model = models.LL_Arm_LSTM()
    else:
        if model_type == 'CNN':
            model = models.Locomotion_CNN(window_size=window_size)
        elif model_type == 'LSTM':
            model = models.Locomotion_LSTM()

    if model_name in os.listdir('trained_model'):
        load_model(model, model_name)
    else:
        with mlflow.start_run():
            log_params_from_omegaconf_dict(config)
            train(model, train_dataloader, val_dataloader, epochs=20)
            test(model, test_dataloader)
        save_model(model, model_name)
    generate_testset_with_prediction_label(config, processor, model, test_dataloader)

if __name__ == "__main__":
    main()