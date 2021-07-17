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
        self.w = np.random.randn(n_features)
        self.b = np.random.randn()

    def get_loss(self, y_hat, y):
        """
        Gets Mean Squared Error between predictions (y_hat) and actual value (y).
        """
        return np.mean((y_hat - y)**2)
    
    def _get_epoch_loss(self, X, y, loss_per_epoch):
        """
        Appends actual loss to loss_per_epoch.
        
        INPUT:  X -> Numeric matrix with features.
                y -> Numeric array with labels.
                loss_per_epoch -> List containing the loss of each epoch.
        
        OUTPUT: loss_per_epoch -> Updated list of losses per epoch.
        """
        
        y_hat = self.predict(X)
        loss = self.get_loss(y_hat, y)
        loss_per_epoch.append(loss)

        return y_hat, loss_per_epoch
    
    def _get_gradients(self, X, y, y_hat):
        """
        Gets the gradients for the Linear Regression parameters
        when optimising them for the given data.

        INPUT:  X -> Matrix of numerical datapoints.
                y -> Target for each row of X.
                y_hat -> Current linear prediction of each target.

        OUTPUT: grad_w -> Gradient of loss with respect to self.w.
                grad_b -> Gradient of loss with respect to self.b.
        """
        error = y_hat - y
        grad_w = 2 * np.mean(np.matmul(error, X), axis=0)
        grad_b = 2 * np.mean(error)
        return grad_w, grad_b
    
    def _update_parameters(self, lr, X_batch, y_batch, y_hat):
        """
        Updates the parameters of the model by substracting the
        product of the learning rate and the gradient of the loss
        with respect to each parameter.
        """
        grad_w, grad_b = self._get_gradients(X_batch, y_batch, y_hat)
        self.w -= lr * grad_w
        self.b -= lr * grad_b


    def predict(self, X):
        """
        Predicts the value of an output for each row of X
        using the fitted Linear Regression model.
        """
        return np.matmul(X, self.w) + self.b

    def fit(self, X, y, X_val, y_val, lr = 0.001, epochs=1000000,
            acceptable_error=0.0001, return_loss=False):
        """
        Optimises the Linear Regression parameters for the given data.

        INPUTS: X -> Matrix of numerical datapoints.
                y -> Target for each row of X.
                lr -> Learning Rate of Mini-batch Gradient Descent.
                        default = 0.001.
                epochs -> Number of iterationns of Mini-Batch Gradient Descent.
                        default = 100
        """
        mean_training_loss = []
        validation_loss = []
        for epoch in range(epochs):
            minibatches = MiniBatch(X, y)
            training_loss_per_epoch = []
            for X_batch, y_batch in minibatches:
                y_hat, training_loss_per_epoch = self._get_epoch_loss(X_batch, y_batch,
                                                    training_loss_per_epoch)
                self._update_parameters(lr, X_batch, y_batch, y_hat)
            
            mean_training_loss.append(np.mean(training_loss_per_epoch))
            if len(X_val) and len(y_val):
                y_hat_val, validation_loss_per_epoch = self._get_epoch_loss(X_val, y_val,
                                                    [])
                validation_loss.append(validation_loss_per_epoch[0])

                if epoch > 2 and abs(validation_loss[-2]- validation_loss[-1]) < acceptable_error:
                    print(f"Validation loss for epoch {epoch} is {validation_loss[-1]}")
                    break

        if return_loss:
            return {'training_set': mean_training_loss,
                    'validation_set': validation_loss}

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

class RidgeRegression(LinearRegression):
    """
    Class representing Ridge Regression predicting model
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
        return np.mean((y_hat - y)**2) + self.rf*sum(self.w**2)
    
    def _get_gradients(self, X, y, y_hat):
        """
        Gets the gradients for the Ridge Regression parameters
        when optimising them for the given data.

        INPUT:  X -> Matrix of numerical datapoints.
                y -> Target for each row of X.
                y_hat -> Current linear prediction of each target.

        OUTPUT: grad_w -> Gradient of loss with respect to self.w.
                grad_b -> Gradient of loss with respect to self.b.
        """
        grad_w, grad_b =  super()._get_gradients(X, y, y_hat)
        grad_w += 2*self.rf*self.w

        return grad_w, grad_b