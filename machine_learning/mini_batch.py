import numpy as np

class MiniBatch:
    """
    Class representing the batching of data.
    """
    def __init__(self, X, y, batchsize=16):
        self.batches = []
        self._get_batches(X, y, batchsize)
    
    def __len__(self):
        return len(self.batches)
    
    def __getitem__(self, idx):
        return self.batches[idx]
    
    def shuffle(self, X, y):
        """ Shuffles the data before dividing it into batches. """
        data = np.c_[X, y]
        np.random.shuffle(data)
        if len(y.shape) > 1:
            num_y = y.shape[1]
            return data[:, :-num_y], data[:, -num_y:]
        else:
            return data[:, :-1], data[:, -1]

    def _get_batches(self, X, y, batchsize):
        """ Divides the pair X, y in random batches of batchsize size. """
        idx = 0
        X, y = self.shuffle(X, y)
        while idx < len(X):
            if idx + batchsize >= len(X):
                self.batches.append(
                    (X[idx:],
                    y[idx:]))
            else:
                self.batches.append(
                    (X[idx:idx+batchsize],
                    y[idx:idx+batchsize]))
            idx += batchsize 