import numpy as np

def sigmoid(x):
    return float(1/(1+np.exp(-x)))

def derivativeSigmoid(x):
    fx = sigmoid(x)
    return fx * (1-fx)
