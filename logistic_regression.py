import numpy as np
from linear_regression import LinearRegression

class BinaryLogisticRegression(LinearRegression):
    """
    Class representing Binary Logistic Regression predicting model
    implemented from the Linear Regression model.
    Only accepts numerical features.
    
    Methods created to match the ones used by Sci-kit Learn models.
    """
    
    def _negative_sigmoid(self, inputs):
        """ Returns the _sigmoid function for negative inputs. """
        exp = np.exp(inputs)
        return exp / (exp + 1)

    def _positive_sigmoid(self, inputs):
        """ Returns the _sigmoid function for positive inputs. """
        return 1 / (1 + np.exp(-inputs))

    def _sigmoid(self, inputs):
        """
        Returns the _sigmoid function of the input.
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
        # y_hat = self._predict(X)
        # return y_hat, -np.mean(y*np.log(y_hat)+(1-y)*np.log(1-y_hat))
        Z = super()._predict(X)
        return self._sigmoid(Z), np.sum(np.mean(np.maximum(Z, 0) - Z*y + np.log(1+np.exp(-abs(Z))), axis=0))
    
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
        
        if len(y.shape) > 1:
            grad_w = np.transpose(np.matmul(np.transpose(dldy*dydz), X))
        else:
            grad_w = np.matmul(dldy*dydz, X)
            
        grad_b = np.sum(dldy*dydz, axis=0)

        return grad_w, grad_b

    def _predict(self, X):
        """
        Predicts the value of an output for each row of X
        using the fitted Logistic Regression model.
        """

        Z = super()._predict(X)
        return self._sigmoid(Z)
    
    def predict(self, X):
        """ Mask that allows to process predicted data before returning it. """
        return round(self._predict(X))

class LogisticRegression(LinearRegression):
    """
    Class representing Multiclass Logistic Regression predicting model
    implemented from the Linear Regression model.
    Only accepts numerical features.
    
    Methods created to match the ones used by Sci-kit Learn models.
    """
    def _softmax(self, inputs):
        """ Returns the softmax function of the inputs. """
        stable_inputs = inputs - np.max(inputs, axis=1)[:, None]
        exps = np.exp(stable_inputs)
        return exps / np.sum(exps, axis=1).reshape(-1, 1)

    def get_loss(self, X, y):
        """
        Gets Cross Entropy between predictions (y_hat) and actual value (y).
        """
        y_hat = self._predict(X)
        return y_hat, np.mean(np.sum(-y*np.log(y_hat), axis=1))

    def _get_gradients(self, X, y, y_hat):
        """
        Gets the gradients for the Logistic Regression parameters
        when optimising them for the given data.

        INPUT:  X -> Matrix of numerical datapoints.
                y -> Target for each row of X.
                y_hat -> Current linear prediction of each target.

        OUTPUT: grad_w -> Gradient of loss with respect to self.w.
                grad_b -> Gradient of loss with respect to self.b.

        For reference:
        https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
        https://deepnotes.io/softmax-crossentropy#derivative-of-softmax
        """
        m = X.shape[0]
        grad = y_hat/m
        grad[range(m), np.argmax(y, axis=1)] -= 1

        grad_w = np.transpose(np.matmul(np.transpose(grad), X))
        grad_b = np.sum(grad, axis=0)

        return grad_w, grad_b

    def _predict(self, X):
        """
        Predicts the value of an output for each row of X
        using the fitted Logistic Regression model.
        """
        Z = super()._predict(X)
        return self._softmax(Z)
    
    def predict(self, X):
        """ Mask that allows to process predicted data before returning it. """
        return np.argmax(self._predict(X), axis=1)

    def _one_hot(self, y):
        """ For a categorical array y, returns a matrix of the one-hot encoded data. """
        m = y.shape[0]
        one_hot = np.zeros((m, np.max(y)+1))
        one_hot[range(m), y] = 1
        return one_hot

    def fit(self, X, y, X_val, y_val, lr=0.001, epochs=1000,
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
        oh_y = self._one_hot(y)
        oh_y_val = self._one_hot(y_val)

        self.__init__(n_features=X.shape[1], n_labels=oh_y.shape[1])
        return super().fit(X=X, y=oh_y, X_val=X_val, y_val=oh_y_val, lr=lr, epochs=epochs,
            acceptable_error=acceptable_error, return_loss=return_loss,
            save_every_epoch=save_every_epoch)