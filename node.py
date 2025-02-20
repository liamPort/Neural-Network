import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class Node:
    def __init__(self, learningRate):    
        self.weights = np.array([0.0, 0.0], dtype=np.float64)
        self.bias = np.float64(0.0)
        self.learningRate = learningRate

    def getWeights(self):
        return self.weights
    
    def getBias(self):
        return self.bias

    def activation(self, x):
        if x >=0:
            return 1
        else:
            return -1

    def predict(self, inputs):
        weightedSum = float(self.bias)
        for i in range(len(inputs)):
            weightedSum += self.weights[i] * inputs[i]
        return self.activation(weightedSum)
