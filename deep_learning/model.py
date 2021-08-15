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

    def fit(self, train_load, test_load=None, X_val=None, y_val=None, lr = 0.001, epochs=1000,
            acceptable_error=0.001, return_loss=False, save_every_epoch=None):
        """
        Optimises the model parameters for the given data.

        INPUTS: train_load -> torch.utils.data.DataLoader object with the data.
                lr -> Learning Rate of Mini-batch Gradient Descent.
                        default = 0.001.
                epochs -> Number of iterationns of Mini-Batch Gradient Descent.
                        default = 100
        """
        optimiser = torch.optim.SGD(self.parameters(), lr=lr)

        mean_training_loss = []
        mean_validation_loss = []

        for epoch in range(epochs):
            training_loss = []
            self.train()
            for X_train, y_train in train_load:
                optimiser.zero_grad()
                y_hat = self.forward(X_train)
                train_loss = self.get_loss(y_hat, y_train)
                training_loss.append(train_loss.item())
                train_loss.backward()
                optimiser.step()
            
            mean_training_loss.append(np.mean(training_loss))

            # if X_val and y_val:
            #     y_hat_val = self.forward(X_val)
            #     validation_loss.append(self.get_loss(y_hat_val, y_val).detach().numpy())

            #     if epoch > 2 and (
            #         (abs(validation_loss[-2]- validation_loss[-1])/validation_loss[-1] < acceptable_error)
            #         or (validation_loss[-1] > validation_loss[-2])):
            #         print(f"Validation train_loss for epoch {epoch} is {validation_loss[-1]}")
            #         break
            
            if test_load:
                validation_loss = []
                self.eval() # set model in inference mode (need this because of dropout)
                for X_val, y_val in test_load:
                    y_hat_val = self.forward(X_val)
                    val_loss = self.get_loss(y_hat_val, y_val)
                    validation_loss.append(val_loss.item())
                mean_validation_loss.append(np.mean(validation_loss))

                if epoch > 2 and (
                    (abs(mean_validation_loss[-2]- mean_validation_loss[-1])/mean_validation_loss[-1] < acceptable_error)
                    or (mean_validation_loss[-1] > mean_validation_loss[-2])):
                    print(f"Validation train_loss for epoch {epoch} is {mean_validation_loss[-1]}")
                    break

        if return_loss:
            return {'training': mean_training_loss,
                    'validation': validation_loss}