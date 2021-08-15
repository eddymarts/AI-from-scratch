import torch
from torch.utils.data import *

class DataSet(Dataset):
  """
  Class that implements torch.utils.data.Dataset
  """
  def __init__(self, X, y, one_hot_target=False,
              normalize=False, split=False, seed=None):
    super().__init__()

    if len(X.shape) > 1:
      self.n_features = X.shape[1]
      self.X = torch.Tensor(X).float()
    else:
      self.n_features = 1
      self.X = torch.Tensor(X.reshape(-1, self.n_features)).float()

    if len(y.shape) > 1:
      self.n_labels = y.shape[1]
      self.y = torch.Tensor(y).float()
    else:
      self.n_labels = 1
      self.y = torch.Tensor(y.reshape(-1, self.n_labels)).float()

    self.mean = torch.mean(self.X, axis=0)
    self.std = torch.std(self.X, axis=0)

    if one_hot_target:
      self.y_oh = self.one_hot(self.y)

    if normalize:
      self.normalize()

    if split:
      self.split(seed)

  def __getitem__(self, idx):
    return self.X[idx], self.y[idx]

  def __len__(self):
      return len(self.X)
  
  def one_hot(self, y):
    """ For a categorical array y, returns a matrix of the one-hot encoded data. """
    m = y.shape[0]
    y = y.long()
    onehot = torch.zeros((m, int(torch.max(y)+1)))
    onehot[range(m), y] = 1
    return onehot

  def normalize(self, data=None):
    if data == None:
      self.X_raw = self.X
      self.X = (self.X - self.mean)/self.std
      return self.X
    else:
      return (data - self.mean)/self.std

  def split(self, seed, sizes=[0.7, 0.15, 0.15], shuffle=True):
    lengths = [round(len(self)*size) for size in sizes]
    lengths[-1] = len(self) - sum(lengths[:-1])

    if seed == None:
      self.splits = random_split(self, lengths)
    else:
      self.splits = random_split(self, lengths,
        generator=torch.Generator().manual_seed(seed))