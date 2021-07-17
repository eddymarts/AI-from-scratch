import matplotlib.pyplot as plt
from scale_split import scalesplit

class GridSearch:
  """
  Class that represents the grid search over parameters of ML models.
  """

  def __init__(self) -> None:
    pass

  def _compare_losses(self, model_class, X, y, parameter):
    X_sets, y_sets = scalesplit(X, y, test_size=0.2, seed=42)

    model = model_class(n_features=X_sets[0].shape[1], **parameter)
    loss = model.fit(X_sets[0], y_sets[0], X_sets[1], y_sets[1], return_loss=True)
    return loss

  def _plot_losses(self, ax, model, parameter, loss):
    ax.plot(
      loss['training_set'], label="Training set loss")
    ax.plot(
      loss['validation_set'], label="Validation set loss")
    ax.set_yticks(np.arange(55, 75, step=5))
    ax.set_ylim([55, 75])
    ax.set(xlabel=f"p={parameter}\nMSE={round(loss['validation_set'][-1])}", ylabel=model.__name__)
    ax.label_outer()
  
  def __call__(self, models, parameters, X, y):
    self.parameters = parameters
    self.models = models

    # Plots the batch loss vs Validation loss
    fig, axs = plt.subplots(len(models), len(parameters), figsize=[10, 5])
    loss = {}
    for m_idx, model in enumerate(models):
      loss[model.__name__] = {}
      for p_idx, parameter in enumerate(parameters):
        p = [p for p in parameter.values()][0]
        loss[model.__name__][p] = self._compare_losses(model, X, y, parameter)
        
        if len(models) > 1:
          self._plot_losses(axs[m_idx, p_idx], model, p, loss[model.__name__][p])
        else:
          self._plot_losses(axs[p_idx], model, p, loss[model.__name__][p])

    plt.savefig(fname="lasso_ridge_rfs.jpg")

if __name__ == "__main__":
  from sklearn import datasets, model_selection
  X, y = datasets.load_boston(return_X_y=True)

  from linear_regression import *
  models = [LassoRegression, RidgeRegression]
  regularization_factors = [{'rf': rf/10} for rf in range(11)]

  grid_search = GridSearch()
  grid_search(models, regularization_factors, X, y)



"""
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
  - visualise this on a X-y graph and play around with the coefficients until you get a function that is not too boring (linear)
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
