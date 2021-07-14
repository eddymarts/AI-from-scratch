import matplotlib.pyplot as plt
import numpy as np
from mini_batch import MiniBatch

class LinearRegression:
    """
    Class representing Linear Regression predicting model
    implemented from scratch.
    Only accepts numerical features.
    
    Methods created to match the ones used by Sci-kit Learn models.
    """
    def __init__(self, n_features) -> None:
        # np.random.seed(42)
        self.w = np.random.randn(n_features)
        self.b = np.random.randn()

    def _get_MSE(self, y_hat, y, rf):
        """
        Gets Mean Squared Error between predictions (y_hat) and actual value (y).
        """
        return np.mean((y_hat - y)**2) + rf*sum(self.w**2)
    
    def _get_validation_loss(self, X_val, y_val, rf):
        y_hat_val = self.predict(X_val)
        validation_loss = self._get_MSE(y_hat_val, y_val, rf)
        return np.mean(validation_loss)
    
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

    def fit(self, X, y, X_val, y_val, lr = 0.001, epochs=1000,
            acceptable_error=0.001, print_loss=False, rf=0.5):
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
        mean_validation_loss = []
        for epoch in range(epochs):
            minibatches = MiniBatch(X, y)
            loss_per_epoch = []
            validation_loss_per_epoch = []
            for X_batch, y_batch in minibatches:
                y_hat = self.predict(X_batch)
                loss = self._get_MSE(y_hat, y_batch, rf)
                grad_w, grad_b = self._get_gradients(X_batch, y_batch, y_hat)
                self.w -= lr * grad_w + 2*rf*self.w
                self.b -= lr * grad_b
                loss_per_epoch.append(loss)
                validation_loss_per_epoch.append(self._get_validation_loss(X_val, y_val, rf))
            mean_loss.append(np.mean(loss_per_epoch))
            mean_validation_loss.append(np.mean(validation_loss_per_epoch))

            if epoch > 2 and abs(mean_validation_loss[-2]- mean_validation_loss[-1]) < acceptable_error:
                print(f"Validation loss for epoch {epoch} is {mean_validation_loss[-1]}")
                break

        if print_loss:
            # Plots the batch loss vs Validation loss
            plt.plot(mean_loss, label="Minibatch loss: L2")
            plt.plot(mean_validation_loss, label="Validation loss: L2")
            plt.legend()
            plt.show()


    def predict(self, X):
        """
        Predicts the value of an output for each row of X
        using the fitted Linear Regression model.
        """
        return np.matmul(X, self.w) + self.b