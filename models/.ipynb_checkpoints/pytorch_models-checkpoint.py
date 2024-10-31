import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

class ModelTrainer:
    def __init__(self, model, loss_fn, optimizer, epochs=100):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        #self.batch_size = batch_size
        self.epochs = epochs
        self.train_losses = []
        self.val_losses = []

    def train(self, train_loader, val_loader):

        for epoch in range(self.epochs):
            self.model.train()
            for batch_x, batch_y in train_loader:
                self.optimizer.zero_grad()
                output = self.model(batch_x)
                loss = self.loss_fn(output, batch_y)
                loss.backward()
                self.optimizer.step()

            val_loss = self.validate(val_loader)
            self.train_losses.append(loss.item())
            self.val_losses.append(val_loss)

            print(f"Epoch [{epoch + 1}/{self.epochs}], Training Loss: {loss.item():.5f}, Validation Loss: {val_loss:.5f}")

    def validate(self, val_loader):
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_x, val_y in val_loader:
                val_y_pred = self.model(val_x)
                val_loss += self.loss_fn(val_y_pred, val_y).item()                
        return val_loss / len(val_loader)

    def predict(self, val_loader):
        self.model.eval()
        val_y_preds = []
        with torch.no_grad():
            for val_x, val_y in val_loader:
                val_y_preds.append(self.model(val_x).numpy())
        return np.squeeze(np.array(val_y_preds))
            
    def plot_results(self, include_training=False):        
        ax=plt.gca()
        res = pd.DataFrame({'epoch': range(1, self.epochs + 1), 'train_loss': self.train_losses, 'val_loss': self.val_losses})
        res.plot(x='epoch',y='val_loss', loglog=True, ylabel='MSE', ax=ax)
        if include_training:
            res.plot(x='epoch',y='train_loss', loglog=True, ylabel='MSE', ax=ax)
        return res, ax


class LinearBaseline(nn.Module):
    def __init__(self, input_size=10, output_size=2):
        super(LinearBaseline, self).__init__()        
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):        
        return self.fc(x)