from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
from scale_split import scalesplit
from grid_search import GridSearch
from linear_regression import MulticlassLogisticRegression

X, y = datasets.load_iris(return_X_y=True)
X_sets, y_sets, n_features, n_labels = scalesplit(X, y, test_size=0.2)

mlr = MulticlassLogisticRegression(n_features=n_features, n_labels=n_labels)
loss = mlr.fit(X_sets[0], y_sets[0], X_sets[1], y_sets[1], return_loss=True,
                epochs=1000000, acceptable_error=0.00001)
grid_search = GridSearch()

fig, ax = plt.subplots(1, 1, figsize=[10, 5])
grid_search.plot_losses(ax, MulticlassLogisticRegression, 0, loss, limits=None)
plt.savefig(fname="multiclass_logistic_regression.jpg")