from sklearn import datasets
import matplotlib.pyplot as plt
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
loss = linear_model.fit(X_train, y_train, X_val, y_val, return_loss=True)
linear_model.predict(X_train)


# Plots the batch loss vs Validation loss
plt.plot(loss['training_set'], label="Training set loss")
plt.plot(loss['validation_set'], label="Validation set loss")
plt.legend()
plt.show()

"""
- implement L1 and L2 regularisation in your from-scratch linear regression code
  - i hope you have made many git commits in this repo before now
  - run your from scratch code and benchmark your current training and validation loss with no regularisaton
  - create a function to compute the scalar L1 norm which takes in your model weights
  - update your loss function to include the penalty
    - dont forget the hyperparameter
  - How do you need to update your gradient calculations?
  - train the model
  - discuss: compare the loss curves and loss values before and after regularizing your model
    - is this what you expected? why?
  - git commit
  - do a grid search over a sensible range of regularisation parameters
- implement early stopping in your from-scratch linear regression code
  - git commit (do not skip this)
  - implement an evaluation of your models generalisation performance on the validation set at the end of every epoch
  - git commit
  - “checkpoint” your model every epoch by saving it
    - create a folder called checkpoints
    - at the start of training, create a folder within the checkpoints folder with the timestamp as the name
    - during training, save each of your model checkpoints here
    - save it with a filename which indicates both at which epoch this was saved and the validation loss it achieved
  - git commit
  - at the end of training for some fixed number of epochs, select the best model and move it to a different folder called best_models
  - all of this folder and file creation should be done programatically, not manually
- sklearn regularisation
  - load the boston dataset
  - normalise it
  - fit the model and evaluate it on the validation set and train set
  - refer to the sklearn docs to see how you regularize a linear regression model
  - regularize the model and compare train and val performance
  - check out the sklearn docs to find a way to grid search over different regularization parameter values
  - try this out with some of the other toy datasets from sklearn
- regularizing a polynomial model’s capacity
  - create a new github repo called Regularization-Experiments
    - you should make at least 3 git commits during this challenge
      - otherwise you are expelled
  - create a (20 by 1) matrix of random x values between 0 and 1
    - these will be our design matrix (20 examples, 1 feature)
  - for the next parts, dont go crazy, just keep it simple
  - defing some function which takes in those single feature examples and returns a new design matrix with an extra column representing the x-squared values
  - generalise this function to be able to return you features which are powers of x up to some number
  - define a function which computes a label such as y = 2 + x + 0.2*x^2 + 0.1*x^2
  - visualise this on a X-Y graph and play around with the coefficients until you get a function that is not too boring (linear)
  - split the data into train, val and test sets
  - fit a model to these labels, firstly just passing your model the original features (x^1)
  - visualise the predictions against the label
    - you should see that the model is underfit
  - now train a series of models on design matrix that contain sequentially increasing powers of x
    - include powers of x way above those which your labels are based on
      - e.g. go up to features where x^12 is included
      - the models trained on these should overfit the data (easy to do if you make the train set small)
  - grid search over the capacity hyperparam (which power of x is included) to evaluate each model on the train and val set
  - dicsuss: what were the results?
"""