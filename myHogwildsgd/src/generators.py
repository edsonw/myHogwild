import numpy as np


class DataGenerator:
    """ Class to generate data for HogWildRegressor. 

    __call__ : generates an iterable of chunks to be processed per epoch
    """
    def __init__(self, shuffle, chunk_size):
        self.shuffle = shuffle
        self.chunk_size = chunk_size

    # when the class ben called
    def __call__(self, X, y):
        # shuffle X and y
        if self.shuffle:
            indices = np.random.choice(len(X), replace=False, size=len(X))
            X = X[indices]
            y = y[indices]
     
        batch_size = int(X.shape[0]/float(self.chunk_size))

        for k in range(batch_size):
            Xx = X[k*self.chunk_size : (k+1)*self.chunk_size]
            yy = y[k*self.chunk_size : (k+1)*self.chunk_size]
            yield (Xx, yy)
