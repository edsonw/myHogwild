import sys
from types import ModuleType
from multiprocessing.sharedctypes import Array
from ctypes import c_double
import numpy as np
import math
temp_module_name = '__hogwildsgd__temp__'
import threading

class SharedWeights:
    """ Class to create a temporary module with the gradient function inside
        to allow multiprocessing to work for async weight updates.
    """
    # 使用 锁确保多线程更新权重不会冲突。
    R=threading.Lock()
    def __init__(self, size_w):
        # create a shared array
        coef_shared = Array(c_double,
                            (np.random.random_sample(size=(size_w, 1)) * 1. / np.sqrt(size_w)).flat,
                            lock=False)

        # read an array from buffer
        w = np.frombuffer(coef_shared)
        # reshape the shared array
        w = w.reshape((len(w), 1))
        self.w = w

    def __enter__(self, *args):
        # Make temporary module to store shared weights

     
        mod = ModuleType(temp_module_name)
        mod.__dict__['w'] = self.w
        sys.modules[mod.__name__] = mod
        self.mod = mod
        return self

    def __exit__(self, *args):
        # Clean up temporary module
        del sys.modules[self.mod.__name__]


# update weight
def gradient_step(X, y, learning_rate):
    R.acquire()
    w = sys.modules[temp_module_name].__dict__['w']
    # caculate gradient
    grad = np.dot(X.reshape(X.shape[1], X.shape[0]),
                  np.multiply(-y, np.exp(np.multiply(-y, np.dot(X, w))) / (1 + np.exp(np.multiply(-y, np.dot(X, w))))))/X.shape[0]
    # update w
    for index in np.where(abs(grad) > .01)[0]:
        w[index] -= learning_rate * grad[index, 0]
    R.release()


# caculate the error
def compute_error(X, y, w):
    err = np.log(1 + np.exp(np.multiply(-y, np.dot(X, w))))
    return np.sum(err)
