import matplotlib.pyplot as plt
import numpy as np
from scale_split import scalesplit

class GridSearch:
  """
  Class that represents the grid search over parameters of ML models.
  """

  def __init__(self) -> None:
    pass

  def compare_losses(self, model_class, X, y, parameter):
    X_sets, y_sets = scalesplit(X, y, test_size=0.2, seed=42)

    model = model_class(n_features=X_sets[0].shape[1], **parameter)
    loss = model.fit(X_sets[0], y_sets[0], X_sets[1], y_sets[1], return_loss=True, save_every_epoch=None)
    return loss

  def plot_losses(self, ax, model, parameter, loss, limits=[55,75]):
    ax.plot(
      loss['training_set'], label="Training set loss")
    ax.plot(
      loss['validation_set'], label="Validation set loss")
    if not(limits is None):
      ax.set_yticks(np.arange(limits[0], limits[1], step=5))
      ax.set_ylim(limits)
    ax.set(xlabel=f"p={parameter}\nl={round(loss['validation_set'][-1], 2)}", ylabel=model.__name__)
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
        loss[model.__name__][p] = self.compare_losses(model, X, y, parameter)
        
        if len(models) > 1:
          self.plot_losses(axs[m_idx, p_idx], model, p, loss[model.__name__][p])
        else:
          self.plot_losses(axs[p_idx], model, p, loss[model.__name__][p])