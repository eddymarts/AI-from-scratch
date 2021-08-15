import torch
from linear_regression import LinearRegression

class NeuralNetworkRegression(LinearRegression):
    def __init__(self, n_features, n_labels):
        super().__init__(n_features, n_labels)
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(n_features, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 30),
            torch.nn.ReLU(),
            torch.nn.Linear(30, 40),
            torch.nn.ReLU(),
            torch.nn.Linear(40, 30),
            torch.nn.ReLU(),
            torch.nn.Linear(30, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, n_labels)
        )