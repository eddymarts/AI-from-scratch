from dataset import DataSet
from linear_regression import LogisticRegression
from sklearn import datasets
from multiprocessing import cpu_count
import torch
from torch.utils.data import *
import matplotlib.pyplot as plt
from sklearn.metrics import *
import torch.nn.functional as F
import numpy as np

X, y = datasets.load_iris(return_X_y=True)

iris_data = DataSet(X, y, normalize=True, split=True)
train_load = DataLoader(iris_data.splits[0], batch_size=16,
            shuffle=True, num_workers=round(cpu_count()/2))

X_val = iris_data.splits[1].dataset.X
y_val = iris_data.splits[1].dataset.y

logistic_regressor = LogisticRegression(iris_data.n_features, int(torch.max(iris_data.y)+1))
loss = logistic_regressor.fit(train_load, X_val, y_val, return_loss=True,
                            epochs=1000, acceptable_error=0.0001, lr=0.001)
y_hat_val = logistic_regressor(X_val)
print(torch.cat((y_val, y_hat_val), dim=1))
print("R^2 score:", r2_score(y_hat_val.detach().numpy(), y))
plt.plot(loss['training'], label="Training set loss")
plt.plot(loss['validation'], label="Validation set loss")
plt.xlabel(f"Epochs\nl={loss['validation'][-1]}")
plt.ylabel("CE")
plt.show()
# plt.savefig(fname="linear_regression.jpg")