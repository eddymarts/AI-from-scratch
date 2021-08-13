from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
from scale_split import scalesplit
from grid_search import GridSearch
from linear_regression import LinearRegression

X, y = datasets.load_boston(return_X_y=True)
y = np.c_[y, y*2]

X_sets, y_sets, n_features, n_labels = scalesplit(X, y, test_size=0.2)

lin_reg = LinearRegression(n_features=n_features, n_labels=n_labels)
loss = lin_reg.fit(X_sets[0], y_sets[0], X_sets[1], y_sets[1], return_loss=True,
                epochs=1000000, acceptable_error=0.00001)

grid_search = GridSearch()

fig, ax = plt.subplots(1, 1, figsize=[10, 5])
grid_search.plot_losses(ax, LinearRegression, 0, loss, limits=None)
plt.savefig(fname="multivariate_regression.jpg")