import torch
import torch.nn.functional as F
import numpy as np

class NeuralNetwork(torch.nn.Module):
    """
    Abstract class for Neural Network model.
    implemented from torch.nn.Module.
    Only accepts numerical features.

    Must implement:
    - layers attribute
    - get_loss method
    
    Methods created to match the ones used by Sci-kit Learn models.
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, X):
        """
        Predicts the value of an output for each row of X
        using the model.
        """
        return self.layers(X)

    def fit(self, data_load,  X_val, y_val, lr = 0.001, epochs=1000,
            acceptable_error=0.001, return_loss=False, save_every_epoch=None):
        """
        Optimises the model parameters for the given data.

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
                y_hat = self.forward(X_train)
                loss = self.get_loss(y_hat, y_train)
                training_loss.append(loss.item())
                loss.backward()
                optimiser.step()
            
            mean_training_loss.append(np.mean(training_loss))

            if len(X_val) and len(y_val):
                y_hat_val = self.forward(X_val)
                validation_loss.append(self.get_loss(y_hat_val, y_val).detach().numpy())

                if epoch > 2 and (
                    (abs(validation_loss[-2]- validation_loss[-1])/validation_loss[-1] < acceptable_error)
                    or (validation_loss[-1] > validation_loss[-2])):
                    print(f"Validation loss for epoch {epoch} is {validation_loss[-1]}")
                    break

        if return_loss:
            return {'training': mean_training_loss,
                    'validation': validation_loss}