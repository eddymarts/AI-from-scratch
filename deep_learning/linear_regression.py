import torch
import torch.nn.functional as F
import numpy as np

class LinearRegression(torch.nn.Module):
    def __init__(self, n_features, n_labels):
        super().__init__()
        self.layers = torch.nn.Linear(n_features, n_labels)
        
    def forward(self, X):
        return self.layers(X)
    
    def get_loss(self, y_hat, y):
        return F.mse_loss(y_hat, y)

    def fit(self, data_load,  X_val, y_val, lr = 0.001, epochs=1000,
            acceptable_error=0.001, return_loss=False, save_every_epoch=None):
        """
        Optimises the Linear Regression parameters for the given data.

        INPUTS: data_load -> torch.utils.data.DataLoader object with the data.
                lr -> Learning Rate of Mini-batch Gradient Descent.
                        default = 0.001.
                epochs -> Number of iterationns of Mini-Batch Gradient Descent.
                        default = 100
        """
        optimiser = torch.optim.SGD(self.parameters(), lr=lr)

        mean_training_loss = []
        validation_loss = []

        for epoch in range(epochs):
            training_loss = []
            for X_train, y_train in data_load:
                optimiser.zero_grad()
                y_hat = self(X_train)
                loss = self.get_loss(y_hat, y_train)
                training_loss.append(loss.item())
                loss.backward()
                optimiser.step()
            
            mean_training_loss.append(np.mean(training_loss))

            if len(X_val) and len(y_val):
                y_hat_val = self(X_val)
                validation_loss.append(self.get_loss(y_hat_val, y_val).detach().numpy())

                if epoch > 2 and (
                    (abs(validation_loss[-2]- validation_loss[-1])/validation_loss[-1] < acceptable_error)
                    or (validation_loss[-1] > validation_loss[-2])):
                    print(f"Validation loss for epoch {epoch} is {validation_loss[-1]}")
                    break

        if return_loss:
            return {'training': mean_training_loss,
                    'validation': validation_loss}

class LogisticRegression(LinearRegression):
    def __init__(self, n_features, n_labels):
        super().__init__(n_features, n_labels)
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(n_features, n_labels), 
            torch.nn.Sigmoid())
    
    def get_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)