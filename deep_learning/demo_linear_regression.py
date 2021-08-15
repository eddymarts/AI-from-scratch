from dataset import DataSet
from linear_models import LinearRegression
from sklearn import datasets
from sklearn.metrics import *
from multiprocessing import cpu_count
import torch
from torch.utils.data import *
import matplotlib.pyplot as plt
import numpy as np

X, y = datasets.load_boston(return_X_y=True)

boston_data = DataSet(X, y, normalize=True, split=True)
train_load = DataLoader(boston_data.splits[0], batch_size=16,
            shuffle=True, num_workers=round(cpu_count()/2))
test_load = DataLoader(boston_data.splits[1], batch_size=16,
            shuffle=False, num_workers=round(cpu_count()/2))

linear_regressor = LinearRegression(boston_data.n_features, boston_data.n_labels)
loss = linear_regressor.fit(train_load, test_load, return_loss=True)

y_val, y_hat_val = linear_regressor.predict(test_load, return_y=True)

print(torch.cat((y_val, y_hat_val), dim=1)[0:10])
print("R^2 score:", r2_score(y_hat_val.detach().numpy(), y_val.detach().numpy()))
plt.plot(loss['training'], label="Training set loss")
plt.plot(loss['validation'], label="Validation set loss")
plt.xlabel(f"Epochs\nl={loss['validation'][-1]}")
plt.ylabel("MSE")
plt.show()
# plt.savefig(fname="linear_regression.jpg")


