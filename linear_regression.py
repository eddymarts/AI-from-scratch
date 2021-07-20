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
    def __init__(self, n_features) -> None:
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

        y_hat = self.predict(X)
        return y_hat, np.mean((y_hat - y)**2)
    
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
        
    def predict(self, X):
        """
        Predicts the value of an output for each row of X
        using the fitted Linear Regression model.
        """
        return np.matmul(X, self.w) + self.b

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

                if epoch > 2 and abs(validation_loss[-2]- validation_loss[-1]) < acceptable_error:
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

    def __init__(self, n_features, rf=0.5) -> None:
        self.rf = rf
        super().__init__(n_features)

    def get_loss(self, X, y):
        """
        Gets Regularised Mean Squared Error between predictions of X and actual value (y).
        
        INPUT:  X -> Numeric matrix with features.
                y -> Numeric array with labels.
        
        OUTPUT: y_hat -> Predictions of labels of X
                loss -> Regularised Mean Squared Error between predictions and actual labels.
        """
        y_hat = self.predict(X)
        return y_hat, np.mean((y_hat - y)**2) + self.rf*sum(abs(self.w))
    
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

    def get_loss(self, X, y):
        """
        Gets Regularised Mean Squared Error between predictions of X and actual value (y).
        
        INPUT:  X -> Numeric matrix with features.
                y -> Numeric array with labels.
        
        OUTPUT: y_hat -> Predictions of labels of X
                loss -> Regularised Mean Squared Error between predictions and actual labels.
        """
        y_hat = self.predict(X)
        return y_hat, np.mean((y_hat - y)**2) + self.rf*sum(self.w**2)
    
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

class BinaryLogisticRegression(LinearRegression):
    """
    Class representing Logistic Regression predicting model
    implemented from the Linear Regression model.
    Only accepts numerical features.
    
    Methods created to match the ones used by Sci-kit Learn models.
    """

    def __init__(self, n_features) -> None:
        super().__init__(n_features)
    
    def _negative_sigmoid(self, inputs):
        """ Returns the sigmoid function for negative inputs. """
        exp = np.exp(inputs)
        return exp / (exp + 1)

    def _positive_sigmoid(self, inputs):
        """ Returns the sigmoid function for positive inputs. """
        return 1 / (1 + np.exp(-inputs))

    def sigmoid(self, inputs):
        """
        Returns the sigmoid function of the input.
        Uses _positive_sigmoid and _negative_sigmoid function depending on the
        sign of the input to avoid computing an arbitrary small or large number
        in the process.
        """

        positive = inputs >= 0
        # Boolean array inversion is faster than another comparison
        negative = ~positive

        # empty contains junk hence will be faster to allocate than zeros
        result = np.empty_like(inputs)
        result[positive] = self._positive_sigmoid(inputs[positive])
        result[negative] = self._negative_sigmoid(inputs[negative])
        return result

    def get_loss(self, X, y):
        """
        Gets Binary Cross Entropy between predictions (y_hat) and actual value (y).
        """
        # y_hat = self.predict(X)
        # return y_hat, -np.mean(y*np.log(y_hat)+(1-y)*np.log(1-y_hat))
        Z = super().predict(X)
        return self.sigmoid(Z), np.mean(np.maximum(Z, 0) - Z*y + np.log(1+np.exp(-abs(Z))))
    
    def _get_gradients(self, X, y, y_hat):
        """
        Gets the gradients for the Logistic Regression parameters
        when optimising them for the given data.

        INPUT:  X -> Matrix of numerical datapoints.
                y -> Target for each row of X.
                y_hat -> Current linear prediction of each target.

        OUTPUT: grad_w -> Gradient of loss with respect to self.w.
                grad_b -> Gradient of loss with respect to self.b.
        """
        
        dldy = -(y/y_hat-(1-y)/(1-y_hat))
        dydz = y_hat*(1-y_hat)/X.shape[0]
        
        grad_w = np.matmul(dldy*dydz, X)
        grad_b = np.dot(dldy, dydz)

        return grad_w, grad_b

    def predict(self, X):
        """
        Predicts the value of an output for each row of X
        using the fitted Logistic Regression model.
        """

        Z = super().predict(X)
        return self.sigmoid(Z)