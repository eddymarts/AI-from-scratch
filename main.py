from sklearn import datasets
import numpy as np
from linear_regression_from_scratch import LinearRegression

X, y = datasets.load_boston(return_X_y=True)
X_mean = np.mean(X, axis=0)
X_sd = np.std(X, axis=0)
X = (X - X_mean)/X_sd

linear_model = LinearRegression(n_features=X.shape[1])
linear_model.fit(X, y, print_loss=True)
linear_model.predict(X)