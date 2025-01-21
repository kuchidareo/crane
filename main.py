import warnings

import hydra
import mlflow
import numpy as np
from omegaconf import OmegaConf
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
    model.eval()
    correct = 0
    total = 0
    for X_batch, y_batch in tqdm(dataloader, desc='Validation'):
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()

    
    mlflow.log_metrics(
        {
            "test_accuracy": correct / total
        }
    )

    print(f'Validation Accuracy: {correct / total:.4f}')

def train(model, dataloader, val_dataloder, epochs=10, patience=5):
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

        epoch_loss = 0
        correct = 0
        total = 0
        for X_batch, y_batch in tqdm(val_dataloder, desc=f'Epoch {epoch + 1}/{epochs}'):
            with torch.no_grad():
                outputs = model(X_batch)
                _, predicted = torch.max(outputs, 1)
                loss = torch.nn.functional.cross_entropy(outputs, y_batch)
                epoch_loss += loss.item()
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
        
        val_loss = epoch_loss / len(val_dataloder)
        mlflow.log_metrics(
            {
                "val_loss": val_loss,
                "val_accuracy": correct / total
            }, step=epoch
        )
        print(f'Epoch {epoch + 1}/{epochs}, Val Loss: {val_loss:.4f}, Val Acc.: {correct / total:.4f}')

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f'Early stopping triggered after {epoch + 1} epochs.')
                break

def save_model(model, label):
    torch.save(model.state_dict(), f'{label}_model.pth')

@hydra.main(config_name="default.yaml", config_path="conf", version_base=None)
def main(config):
    print(OmegaConf.to_yaml(config))
    mlflow.set_experiment(config.experiment_name)

    processor = DataProcessor()
    df = processor.read_data()
   
    window_size = config.window_size
    model_type = config.model_type
    position = config.position

    sensors = processor.position_to_sensor[position]
    columns = [processor.sensor_to_column[sensor] for sensor in sensors]
    flatten_columns = [item for row in columns for item in row]
    label_column = processor.position_to_label[position]

    X = np.asarray(df[flatten_columns])
    y = np.asarray(df[label_column])

    X, y = processor.transform_to_window_data(X, y, window_size=window_size)

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.20)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, shuffle=False, test_size=0.5)
    train_dataloader = create_dataloader(X_train, y_train, batch_size=32, shuffle=True)
    val_dataloder = create_dataloader(X_val, y_val, batch_size=32, shuffle=False)
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
    
    with mlflow.start_run():
        log_params_from_omegaconf_dict(config)
        train(model, train_dataloader, val_dataloder, epochs=20)
        test(model, test_dataloader)
    # save_model(model, position)

if __name__ == "__main__":
    main()