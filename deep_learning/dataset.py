import torch

class Dataset(torch.utils.data.Dataset):
  def __init__(self, X, y, normalize=False):
    super().__init__()
    self.normalize = normalize

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

  def __getitem__(self, idx):
    if self.normalize:
      X = (self.X[idx] - self.mean)/self.std
    else:
      X = self.X[idx]

    return X, self.y[idx]

  def __len__(self):
      return len(self.X)