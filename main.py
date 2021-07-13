from sklearn import datasets
import numpy as np
from linear_regression_from_scratch import LinearRegression

def ScaleSplit(X, Y, size):
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = size, shuffle=True)
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size = size, shuffle=True)
    
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X_train)
    X_train_sc = sc.transform(X_train)
    X_test_sc = sc.transform(X_test)
    X_val_sc = sc.transform(X_val)

    return X_train_sc, X_test_sc, X_val_sc, Y_train, Y_test, Y_val

X, y = datasets.load_boston(return_X_y=True)

X_train, X_test, X_val, y_train, y_test, y_val = ScaleSplit(X, y, size=0.2)

linear_model = LinearRegression(n_features=X_train.shape[1])
linear_model.fit(X_train, y_train, X_val, y_val, print_loss=True)
linear_model.predict(X_train)