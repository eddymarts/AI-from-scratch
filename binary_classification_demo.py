from sklearn import datasets
import matplotlib.pyplot as plt
from scale_split import scalesplit
from grid_search import GridSearch
from linear_regression import BinaryLogisticRegression

X, y = datasets.load_breast_cancer(return_X_y=True)
X_sets, y_sets = scalesplit(X, y, test_size=0.2)

blr = BinaryLogisticRegression(n_features=X_sets[0].shape[1])
loss = blr.fit(X_sets[0], y_sets[0], X_sets[1], y_sets[1], return_loss=True, save_every_epoch=None)
grid_search = GridSearch()

fig, ax = plt.subplots(1, 1, figsize=[10, 5])
grid_search.plot_losses(ax, BinaryLogisticRegression, 0, loss, limits=None)
plt.savefig(fname="binary_logistic_regression.jpg")