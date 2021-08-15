import torch
import torch.nn.functional as F
import numpy as np
from model import NeuralNetwork

class LinearRegression(NeuralNetwork):
    """
    Class representing Linear Regression predicting model
    implemented from Abstract class NeuralNetwork.
    Only accepts numerical features.
    
    Methods created to match the ones used by Sci-kit Learn models.
    """
    def __init__(self, n_features, n_labels):
        super().__init__()
        self.layers = torch.nn.Linear(n_features, n_labels)
    
    def get_loss(self, y_hat, y):
        """
        Gets Mean Squared Error between predictions of X and actual value (y).
        
        INPUT:  y_hat -> Tensor with predicted values.
                y -> Tensor with labels.
        
        OUTPUT: loss -> Mean Squared Error between predictions and actual labels.
        """
        return F.mse_loss(y_hat, y)

class BinaryLogisticRegression(LinearRegression):
    """
    Class representing Binary Logistic Regression predicting model
    implemented from the Linear Regression model.
    Only accepts numerical features.
    
    Methods created to match the ones used by Sci-kit Learn models.
    """
    def __init__(self, n_features, n_labels):
        super().__init__(n_features, n_labels)
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(n_features, n_labels), 
            torch.nn.Sigmoid())
    
    def __call__(self, X):
        """
        Predicts the value of an output for each row of X
        using the Binary Logistic Regression model.
        """
        return torch.round(self.forward(X))
    
    def get_loss(self, y_hat, y):
        """
        Gets Binary Cross Entropy between predictions (y_hat) and actual value (y).
        """
        return F.binary_cross_entropy(y_hat, y)

class LogisticRegression(BinaryLogisticRegression):
    """
    Class representing Multiclass Logistic Regression predicting model
    implemented from the Linear Regression model.
    Only accepts numerical features.
    
    Methods created to match the ones used by Sci-kit Learn models.
    """
    def __init__(self, n_features, n_labels):
        super().__init__(n_features, n_labels)
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(n_features, n_labels), 
            torch.nn.Softmax(1))
    
    def __call__(self, X):
        """
        Predicts the value of an output for each row of X
        using the Logistic Regression model.
        """
        return torch.argmax(self.forward(X), axis=1).reshape(-1, 1)
    
    def get_loss(self, y_hat, y):
        """
        Gets Cross Entropy between predictions (y_hat) and actual value (y).
        """
        return F.cross_entropy(y_hat, y.long().reshape(-1))