import numpy as np
from linear_regression_from_scratch import LinearRegression

class LassoRegression(LinearRegression):
    """
    Class representing Lasso Regression predicting model
    implemented from the Linear Regression model.
    Only accepts numerical features.
    
    Methods created to match the ones used by Sci-kit Learn models.
    """

    def __init__(self, n_features, rf=0.5) -> None:
        self.rf = rf
        super().__init__(n_features)

    def get_loss(self, y_hat, y):
        """
        Gets Mean Squared Error between predictions (y_hat) and actual value (y).
        """
        return np.mean((y_hat - y)**2) + self.rf*sum(abs(self.w))
    
    def _get_gradients(self, X, y, y_hat):
        """
        Gets the gradients for the Lasso Regression parameters
        when optimising them for the given data.

        INPUT:  X -> Matrix of numerical datapoints.
                y -> Target for each row of X.
                y_hat -> Current linear prediction of each target.

        OUTPUT: grad_w -> Gradient of loss with respect to self.w.
                grad_b -> Gradient of loss with respect to self.b.
        """
        grad_w, grad_b =  super()._get_gradients(X, y, y_hat)
        grad_w += self.rf*np.sign(self.w)

        return grad_w, grad_b