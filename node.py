import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random

class Node:
    def __init__(self, inputSize): 
        #caculates the range a weight can take based on input size
        weightRange = 2/inputSize
        self.bias = random.uniform(-weightRange, weightRange)
        self.weights = []
        for i in range(inputSize):
            self.weights.append(random.uniform(-weightRange, weightRange))
        
    def printNodeInfo(self, inputString):
        print(f"\n-------Node{inputString}--------")
        print(f"Num Of Inputs: {len(self.weights)}")
        print(f"Bias: {self.bias}")
        print(f"Weights: {self.weights}")

