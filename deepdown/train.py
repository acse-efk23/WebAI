import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from livelossplot import PlotLosses

from deepdown.visualise import Plot3D, PlotLoss
from deepdown import *


def mse_loss(prediction, target):
        return torch.mean((prediction - target) ** 2)


def r2_score(prediction, target):
    ss_res = torch.sum((target - prediction) ** 2)
    ss_tot = torch.sum((target - torch.mean(target)) ** 2)
    return 1 - ss_res / ss_tot


def r2_score_numpy(prediction, target):
    ss_res = np.sum((target - prediction) ** 2)
    ss_tot = np.sum((target - np.mean(target)) ** 2)
    return 1 - ss_res / ss_tot


def rae_score_numpy(prediction, target):
    nominator = np.sum(np.abs(prediction - target))
    denominator = np.sum(np.abs(target - np.mean(target)))
    return nominator / denominator


class MLOperator:
    def __init__(self):
        self.train_dataset = None
        self.validation_dataset = None
        self.LEARNING_RATE = 0.001
        self.BATCH_SIZE = 8
        self.EPOCHS = 10
        self.current_epochs = 0
        
        self.train_loader = None
        self.validation_loader = None
        self.model = None
        self.optimizer = None
        self.mse_loss = nn.MSELoss()
        self.r2_score = r2_score

        self.history = None

        # Prediction
        self.prediction = None

        # Metadata
        self.features = []
        self.time_steps = None
        self.grid_shape = None


    def initialise(self, train_dataset, validation_dataset, batch_size=8, epochs=10):
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.BATCH_SIZE = batch_size
        self.EPOCHS = epochs
        self.history = {
            'train_loss': [],
            'train_r2': [],
            'validation_loss': [],
            'validation_r2': [],
            'current_epochs': 0
        }
        self.current_epochs = 0

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.BATCH_SIZE, shuffle=True)
        self.validation_loader = DataLoader(self.validation_dataset, batch_size=len(self.validation_dataset), shuffle=False)
        self.model = ResNet(in_channels=self.train_dataset[0][0].shape[0], out_channels=self.train_dataset[0][1].shape[0])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.LEARNING_RATE)

    
    def train(self, epochs=0):

        if epochs:
            self.EPOCHS = epochs

        liveloss = PlotLosses()
        for _ in range(self.EPOCHS):
            
            train_loss, train_r2 = self.train_one_epoch()
            validation_loss, validation_r2 = self.validate_one_epoch()

            logs = {}
            logs['log loss'] = train_loss
            logs['val_log loss'] = validation_loss
            logs['acc'] = train_r2
            logs['val_acc'] = validation_r2
            liveloss.update(logs)
            liveloss.draw()


    def train_visual(self, epochs=5):

        for _ in range(epochs):

            train_loss, train_r2 = self.train_one_epoch()
            validation_loss, validation_r2 = self.validate_one_epoch()

            # Store the results
            self.current_epochs += 1
            self.history["train_loss"].append(train_loss)
            self.history["train_r2"].append(train_r2)
            self.history["validation_loss"].append(validation_loss)
            self.history["validation_r2"].append(validation_r2)
            self.history["current_epochs"] = self.current_epochs

        # Create plotly figures
        x, y = self.train_dataset[0]
        y_pred = self.predict(x.unsqueeze(0))[0][-1]

        return PlotLoss(self.history), Plot3D(y_pred.detach().numpy(), options={'filter': 2, 'sample': 0.5}, title='Predicted Pressure Field')
            
        
    def train_one_epoch(self):
        self.model.train()
        batch_loss = 0
        batch_r2 = 0
        for x, y in self.train_loader:
            self.optimizer.zero_grad()
            prediction = self.model(x)
            loss = self.mse_loss(prediction, y)
            loss.backward()
            self.optimizer.step()
            batch_loss += loss.item() * x.size(0)
            batch_r2 += self.r2_score(prediction, y).item() * x.size(0)
        return batch_loss / len(self.train_dataset), batch_r2 / len(self.train_dataset)


    def validate_one_epoch(self):
        self.model.eval()
        batch_loss = 0
        batch_r2 = 0
        with torch.no_grad():
            for x, y in self.validation_loader:
                prediction = self.model(x)
                batch_loss += self.mse_loss(prediction, y).item() * x.size(0)
                batch_r2 += self.r2_score(prediction, y).item() * x.size(0)
        return batch_loss / len(self.validation_dataset), batch_r2 / len(self.validation_dataset)


    def save_model(self, path='model.pth'):
        torch.save(self.model.state_dict(), path)
        print(f'Model saved to {path}')


    def load_model(self, path='model.pth'):
        self.model.load_state_dict(torch.load(path))
        print(f'Model loaded from {path}')


    def load_json(self, path='dict.json'):
        if os.path.exists(path):
            with open(path, 'r') as file:
                data = json.load(file)
                self.features = data.get('static_properties', [])
                self.time_steps = data.get('time_steps', [])
                self.grid_shape = data.get('grid_size', [])
                print(f'Loaded features: {self.features}, time steps: {self.time_steps}, grid shape: {self.grid_shape}')


    def predict(self, x):
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(x)
        return prediction