# Author: Andrea Vaiuso
# Version: 1.0
# Date: 21.07.2025
# Description: This module defines the RotorModel class for predicting rotor aerodynamic coefficients using a neural network.

import torch
import torch.nn as nn
import torch.optim as optim

class RotorModel(nn.Module):
    def __init__(self, n_inputs=1, n_outputs=6, norm_params_path='normalization_params.pth'):
        super(RotorModel, self).__init__()
        self.fc1 = nn.Linear(n_inputs, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, n_outputs)
        self.norm_params_path = norm_params_path
        
        # Load normalization parameters if they exist
        try:
            params = torch.load(self.norm_params_path)
            self.input_mean = params['input_mean']
            self.input_std = params['input_std']
            self.output_mean = params['output_mean']
            self.output_std = params['output_std']
        except FileNotFoundError:
            self.input_mean = None
            self.input_std = None
            self.output_mean = None
            self.output_std = None


    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def normalize(self, X, mean, std):
        return (X - mean) / std

    def denormalize(self, X, mean, std):
        return X * std + mean

    def train_model(self, X_train: torch.Tensor, y_train: torch.Tensor,
                    X_val: torch.Tensor, y_val: torch.Tensor,
                    epochs: int = 1000, lr: float = 0.001, early_stopping_patience: int = 50):
        # Compute normalization parameters if not already set
        if self.input_mean is None or self.input_std is None or self.output_mean is None or self.output_std is None:
            self.input_mean = X_train.mean(dim=0, keepdim=True)
            self.input_std = X_train.std(dim=0, keepdim=True) + 1e-8
            self.output_mean = y_train.mean(dim=0, keepdim=True)
            self.output_std = y_train.std(dim=0, keepdim=True) + 1e-8
            # Save normalization parameters
            torch.save({
                'input_mean': self.input_mean,
                'input_std': self.input_std,
                'output_mean': self.output_mean,
                'output_std': self.output_std
            }, self.norm_params_path)

        # Normalize data
        X_train_norm = self.normalize(X_train, self.input_mean, self.input_std)
        y_train_norm = self.normalize(y_train, self.output_mean, self.output_std)
        X_val_norm = self.normalize(X_val, self.input_mean, self.input_std)
        y_val_norm = self.normalize(y_val, self.output_mean, self.output_std)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        best_val_loss = float('inf')
        patience_counter = 0
        best_model = None
        for epoch in range(epochs):
            self.train()
            optimizer.zero_grad()
            outputs = self(X_train_norm)
            loss = criterion(outputs, y_train_norm)
            loss.backward()
            optimizer.step()
            self.eval()
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.8f}, Val Loss: {best_val_loss:.8f}', end='\r')
            with torch.no_grad():
                val_outputs = self(X_val_norm)
                val_loss = criterion(val_outputs, y_val_norm)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = self.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f'Early stopping at epoch {epoch + 1}')
                break
        if best_model is not None:
            self.load_state_dict(best_model)
            print(f'Best validation loss: {best_val_loss.item()} at epoch {epoch + 1}')

    def predict(self, X: torch.Tensor):
        self.eval()
        with torch.no_grad():
            # Normalize input
            X_norm = self.normalize(X, self.input_mean, self.input_std)
            y_norm = self(X_norm)
            # Denormalize output
            y = self.denormalize(y_norm, self.output_mean, self.output_std)
            return y
        
    def predict_aerodynamic(self, rpm: float):
        """
        Predict aerodynamic coefficients for a given RPM.
        Args:
            rpm (float): The RPM value to predict aerodynamic coefficients for.
        Returns:
            tuple: A tuple containing the predicted values (T, Q, P, CT, CQ, CP).
        """
        self.eval()
        output = self.predict(torch.tensor([[rpm]], dtype=torch.float32))
        T = output[0][0].item()
        Q = output[0][1].item()
        P = output[0][2].item()
        CT = output[0][3].item()
        CQ = output[0][4].item()
        CP = output[0][5].item()
        # Replace negative values with zero
        return max(T, 0.0), max(Q, 0.0), max(P, 0.0), max(CT, 0.0), max(CQ, 0.0), max(CP, 0.0)

    def save_model(self, filename: str):
        """
        Save the model state dictionary to a file.
        Args:
            filename (str): The filename to save the model state dictionary.
        """
        torch.save(self.state_dict(), filename)
    
    def load_model(self, filename: str):
        """
        Load the model state dictionary from a file.
        Args:
            filename (str): The filename to load the model state dictionary from.
        """
        self.load_state_dict(torch.load(filename))