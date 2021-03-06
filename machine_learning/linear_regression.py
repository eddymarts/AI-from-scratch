import numpy as np
from datetime import datetime
from pathlib import Path
from mini_batch import MiniBatch

class LinearRegression:
    """
    Class representing Linear Regression predicting model
    implemented from scratch.
    Only accepts numerical features.
    
    Methods created to match the ones used by Sci-kit Learn models.
    """
    def __init__(self, n_features, n_labels) -> None:
        if n_labels > 1:
            self.w = np.random.randn(n_features, n_labels)
            self.b = np.random.randn(1, n_labels)
        else:
            self.w = np.random.randn(n_features)
            self.b = np.random.randn()

    def get_loss(self, X, y):
        """
        Gets Mean Squared Error between predictions of X and actual value (y).
        
        INPUT:  X -> Numeric matrix with features.
                y -> Numeric array with labels.
        
        OUTPUT: y_hat -> Predictions of labels of X
                loss -> Mean Squared Error between predictions and actual labels.
        """
        
        y_hat = self._predict(X)
        return y_hat, np.sum(np.mean((y_hat - y)**2, axis=0))
    
    def _get_epoch_loss(self, X, y, loss_per_epoch):
        """
        Appends actual loss to loss_per_epoch.
        
        INPUT:  X -> Numeric matrix with features.
                y -> Numeric array with labels.
                loss_per_epoch -> List containing the loss of each epoch.
        
        OUTPUT: loss_per_epoch -> Updated list of losses per epoch.
        """
        
        y_hat, loss = self.get_loss(X, y)
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
        error = (y_hat - y)/X.shape[0]
        grad_b = 2 * np.sum(error, axis=0)

        if len(y.shape) > 1:
            grad_w = 2 * np.transpose(np.matmul(np.transpose(error), X))
        else:
            grad_w = 2 * np.matmul(error, X)

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
    
    def _file_for_model(self):
        """ Returns file path to store parameters of model. """
        now = datetime.now().strftime("%Y%m%d%H%M%S%f")
        path = f"./checkpoints/{type(self).__name__}/{now}/"
        Path(path).mkdir(parents=True, exist_ok=True)
        return path
        
    def save_params(self, file=None, epoch=None, X=None, y=None):
        """ Saves parameters of model in specified file. """
        if file is None:
            file = self._file_for_model() + "params.txt"
        
        if not(X is None or y is None):
            y_hat, loss = self._get_epoch_loss(X, y, [])
            loss = str(loss[0])
        else:
            loss = ""

        if epoch is None:
            with open(file, "a") as f:
                f.write(f"w={self.w.tostring()}\nb={self.b.tostring()}")
        else:
            with open(file, "a") as f:
                f.write(f"EPOCH {epoch}\nLoss={loss}\nw={self.w.tobytes()}\nb={self.b.tostring()}\n\n")

    def _get_value(self, model, value, type):
        """ Gets desired value from models' file. """
        return [type(val.split("\n")[0]) for val in model.split(f"{value}=")[1:]]
    
    def get_best(self, dir, file, adjust=True):
        """ Get parameters with minimum loss from file. """
        with open(file, 'r') as f:
            model = f.read()
        
        losses = self._get_value(model, "Loss", float)
        index = losses.index(min(losses))
        loss = losses[index]
        w = self._get_value(model, "w", str)[index]
        b = self._get_value(model, "b", str)

        print(np.fromstring(w, dtype=float))

        best = dir + "best.txt"
        
        # with open(best, "a") as f:
        #     f.write(f"Loss={loss}\nw={str(w)}\nb={str(b)}\n\n")
        
        # if adjust:
        #     self.w = w
        #     self.b = w
        
    def _predict(self, X):
        """
        Predicts the value of an output for each row of X
        using the fitted Linear Regression model.
        """
        return np.matmul(X, self.w) + self.b
    
    def predict(self, X):
        """
        Mask that allows to process predicted data before returning it
        in derivated classes.
        """
        return self._predict(X)

    def fit(self, X, y, X_val, y_val, lr = 0.001, epochs=1000,
            acceptable_error=0.001, return_loss=False, save_every_epoch=None):
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

        if not (save_every_epoch is None):
            dir = self._file_for_model()
            file = dir + "params.txt"

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

                if epoch > 2 and (
                    (abs(validation_loss[-2]- validation_loss[-1])/validation_loss[-1] < acceptable_error)
                    or (validation_loss[-1] > validation_loss[-2])):
                    print(f"Validation loss for epoch {epoch} is {validation_loss[-1]}")
                    break
            
            if not (save_every_epoch is None):
                if epoch % save_every_epoch == 0:
                    self.save_params(file=file, epoch=epoch, X=X_val, y=y_val)

        if not (save_every_epoch is None):
            self.get_best(dir, file)

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

    def __init__(self, n_features, n_labels, rf=0.5) -> None:
        self.rf = rf
        super().__init__(n_features, n_labels)

    def get_loss(self, X, y):
        """
        Gets Regularised Mean Squared Error between predictions of X and actual value (y).
        
        INPUT:  X -> Numeric matrix with features.
                y -> Numeric array with labels.
        
        OUTPUT: y_hat -> Predictions of labels of X
                loss -> Regularised Mean Squared Error between predictions and actual labels.
        """
        y_hat, loss = super().get_loss(X, y)
        return y_hat, loss + self.rf*np.sum(np.sum(abs(self.w), axis=0))
    
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

    def __init__(self, n_features, n_labels, rf=0.5) -> None:
        self.rf = rf
        super().__init__(n_features, n_labels)

    def get_loss(self, X, y):
        """
        Gets Regularised Mean Squared Error between predictions of X and actual value (y).
        
        INPUT:  X -> Numeric matrix with features.
                y -> Numeric array with labels.
        
        OUTPUT: y_hat -> Predictions of labels of X
                loss -> Regularised Mean Squared Error between predictions and actual labels.
        """
        y_hat, loss = super().get_loss(X, y)
        return y_hat, loss + self.rf*np.sum(np.sum(self.w**2, axis=0))
    
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