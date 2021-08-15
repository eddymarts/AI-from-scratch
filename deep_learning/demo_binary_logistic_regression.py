from dataset import DataSet
from linear_models import BinaryLogisticRegression
from sklearn import datasets
from sklearn.metrics import *
from multiprocessing import cpu_count
import torch
from torch.utils.data import *
import matplotlib.pyplot as plt
import numpy as np

X, y = datasets.load_breast_cancer(return_X_y=True)

breast_cancer_data = DataSet(X, y, normalize=True, split=True)
train_load = DataLoader(breast_cancer_data.splits[0], batch_size=16,
            shuffle=True, num_workers=round(cpu_count()/2))
test_load = DataLoader(breast_cancer_data.splits[1], batch_size=16,
            shuffle=False, num_workers=round(cpu_count()/2))

logistic_regressor = BinaryLogisticRegression(breast_cancer_data.n_features, breast_cancer_data.n_labels)
loss = logistic_regressor.fit(train_load, test_load, return_loss=True)
y_val, y_hat_val = logistic_regressor.predict(test_load, return_y=True)

print(torch.cat((y_val, y_hat_val), dim=1)[0:10])
print("R^2 score:", r2_score(y_hat_val.detach().numpy(), y_val.detach().numpy()))
plt.plot(loss['training'], label="Training set loss")
plt.plot(loss['validation'], label="Validation set loss")
plt.xlabel(f"Epochs\nl={loss['validation'][-1]}")
plt.ylabel("BCE")
plt.show()


