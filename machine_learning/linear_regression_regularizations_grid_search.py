import matplotlib.pyplot as plt
from grid_search import GridSearch
from sklearn import datasets
from linear_regression import LassoRegression, RidgeRegression

X, y = datasets.load_boston(return_X_y=True)

models = [LassoRegression, RidgeRegression]
regularization_factors = [{'rf': rf/10} for rf in range(11)]

grid_search = GridSearch()
grid_search(models, regularization_factors, X, y)
plt.savefig(fname="lasso_ridge_rfs.jpg")