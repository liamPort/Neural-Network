import numpy as np

def sigmoid(x):
    return float(1/(1+np.exp(-x)))

def derivativeSigmoid(x):
    fx = sigmoid(x)
    return fx * (1-fx)

def mseCalculation(predictions, targets):
    totalError = 0
    length = len(predictions)

    for i in range(length):
        totalError += (predictions[i] - targets[i]) ** 2
    
    return totalError / length
