import unittest
from hogwildsgd import HogWildRegressor
import scipy.sparse
import numpy as np


class TestHogwild(unittest.TestCase):

    """
    loadData:loadData form input/w8a.txt

    test_work: set the parameters and test hogwildsgd! first fit the X,y then predict the X and caculate Accuracy
    """

    def loadData(self):
        X = np.zeros((59245, 300), int)
        y = np.zeros((59245, 1), int)
        index = 0
        with open('../input/w8a.txt','r') as file:
            for line in file:
                item = line.split(" ")
                y[index] = int(item[0])
                for i in range(1, len(item)-1):
                    splitedPair = item[i].split(":")
                    X[index][int(splitedPair[0]) - 1] = 1
                index += 1
        return X, y

    def test_work(self):

        iterations = 100
        step = 0.005
        decay = (step / 2) / iterations
        batch_size = 500
        n_jobs = 4

        custom = input("Enter 'y/n' to choose whether to change the default parameters\n")
        if custom == 'y':
            iterations = int(input("Enter Iterations: "))
            print()
            step = float(input("Enter Step size: "))
            print()
            decay = float(input("Enter Decay: "))
            print()
            n_jobs = int(input("Enter Max Threads: "))
            print()
            batch_size = int(input("Enter Batch: "))
            print()

        X, y = self.loadData()

        hw = HogWildRegressor(n_jobs = n_jobs,
                              iterations = iterations,
                              batch_size = batch_size,
                              step_size = step,
                              decay = decay,
                              chunk_size = 14812)

        hw = hw.fit(X,y)

        y_hat = hw.predict(X)
        y = y.reshape((len(y),))
        
        count = 0
        for i in range(len(y)):
            if y_hat[i] < 0 and y[i] < 0:
                count += 1
            elif y_hat[i] > 0 and y[i] > 0:
                count += 1

        print(count / len(y))


if __name__ == '__main__':
    unittest.main()





