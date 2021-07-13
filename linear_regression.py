from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

class DataLoader:
    def __init__(self, X, y, batchsize=16):
        self.batches = []
        self._get_batches(X, y, batchsize)
    
    def __len__(self):
        return len(self.batches)
    
    def __getitem__(self, idx):
        return self.batches[idx]
    
    def _shuffle(self, X, y):
        """ Shuffles the data before dividing it into batches. """
        data = np.c_[X, y]
        np.random.shuffle(data)
        return data[:, :-1], data[:, -1]

    def _get_batches(self, X, y, batchsize):
        """ Divides the pair X, y in random batches of batchsize size. """
        idx = 0
        X, y = self._shuffle(X, y)
        while idx < len(X):
            if idx + batchsize >= len(X):
                self.batches.append(
                    (X[idx:],
                    y[idx:]))
            else:
                self.batches.append(
                    (X[idx:idx+batchsize],
                    y[idx:idx+batchsize]))
            idx += batchsize 

class LinearRegression:
    def __init__(self, n_features) -> None:
        self.w = np.random.randn(n_features)
        self.b = np.random.randn()

    def _get_MSE(self, y_hat, y):
        """
        Gets Mean Squared Error between predictions (y_hat) and actual value (y).
        """
        return np.mean((y_hat - y)**2)
    
    def _get_gradients(self, X, y, y_hat):
        """
        Gets the gradients for the Linear Regression parameters
        when optimising them for the given data.

        INPUTS: X -> Matrix of numerical datapoints.
                y -> Target for each row of X.
                y_hat -> Current linear prediction of each target.
        """
        error = y_hat - y
        grad_w = 2 * np.mean(np.matmul(error, X), axis=0)
        grad_b = 2 * np.mean(error)
        return grad_w, grad_b

    def fit(self, X, y, lr = 0.001, epochs=100, print_loss=False):
        """
        Optimises the Linear Regression parameters for the given data.

        INPUTS: X -> Matrix of numerical datapoints.
                y -> Target for each row of X.
                lr -> Learning Rate of Mini-batch Gradient Descent.
                        default = 0.001.
                epochs -> Number of iterationns of Mini-Batch Gradient Descent.
                        default = 100
        """
        mean_loss = []
        for epoch in range(epochs):
            data_loader = DataLoader(X, y)
            loss_per_epoch = []
            for X_batch, y_batch in data_loader:
                y_hat = self.predict(X_batch)
                loss = self._get_MSE(y_hat, y_batch)
                grad_w, grad_b = self._get_gradients(X_batch, y_batch, y_hat)
                self.w -= lr * grad_w
                self.b -= lr * grad_b
                loss_per_epoch.append(loss)
            mean_loss.append(np.mean(loss_per_epoch))

        if print_loss:
            plt.plot(mean_loss)
            plt.show()

    def predict(self, X):
        """
        Predicts the value of an output for each row of X
        using the fitted Linear Regression model.
        """
        return np.matmul(X, self.w) + self.b

X, y = datasets.load_boston(return_X_y=True)
X_mean = np.mean(X, axis=0)
X_sd = np.std(X, axis=0)
X = (X - X_mean)/X_sd

linear_model = LinearRegression(n_features=X.shape[1])
linear_model.fit(X, y, print_loss=True)
linear_model.predict(X)