import numpy as np
from sklearn.externals.joblib import Parallel, delayed
from shared import SharedWeights,gradient_step,compute_error
from generators import DataGenerator
import time


class HogWildRegressor():
    def __init__(self,n_jobs=-1,iterations=5,batch_size=1,chunk_size=32,step_size=.001,decay=0.00001):

        self.batch_size = batch_size
        self.n_jobs = n_jobs
        self.iterations = iterations
        self.step_size = step_size
        self.decay = decay
        self.chunk_size = chunk_size

        self.shared_weights = SharedWeights
        self.generator = DataGenerator(shuffle=True, chunk_size=self.chunk_size)
        self.gradient = gradient_step
        # save the updated weighted
        self.coef_ = 0

    def fit(self, X, y):
        start_time = time.time()
        np.random.seed(2018)
        y = y.reshape((len(y), 1))
        size_w = X.shape[1]

        with self.shared_weights(size_w=size_w) as sw:
            for iteration in range(self.iterations):

                Parallel(n_jobs=self.n_jobs)(delayed(self.train_epoch)(e) for e in self.generator(X, y))
                self.step_size -= self.decay
                print("Iteration number: {0}, Error: {1}, Elapsed time: {2}".format(iteration, format(
                    compute_error(X, y, sw.w), '.2f'), format(time.time() - start_time, '.2f')))

        self.coef_ = sw.w

        return self

    def train_epoch(self, inputs):
        X, y = inputs
        batch_size = self.batch_size
        for k in range(int(X.shape[0] / float(batch_size))):
            Xx = X[k * batch_size: (k + 1) * batch_size, :]
            yy = y[k * batch_size: (k + 1) * batch_size]
            self.gradient(Xx, yy, self.step_size)

    def predict(self, X):
        return np.dot(X, self.coef_)




