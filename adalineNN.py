import pandas as pd
import neuralNetwok
import random

class adalineNN(neuralNetwok.NeuralNetwork):
    def __init__(self, trainingDf, learningRate):
        super().__init__(trainingDf, learningRate)
        node = self.node
        node.bias = random.uniform(-1, 1)
        for i in range(len(node.weights)):
            node.weights[i] = random.uniform(-1, 1)
        
    
    def predict(self, inputs):
        node = self.node
        weightedSum = float(node.bias)
        for i in range(len(inputs)):
            weightedSum += node.weights[i] * inputs[i]
        return weightedSum
    
    def learn(self, epochs):
        for epoch in range(epochs):
            toatalError = 0
            for inputs, target in self.zippedDf:
                prediction = self.predict(inputs)
                error = target - prediction
                toatalError += error
                self.updateNeuron(error, inputs)
        
            if(epoch % 100 == 0):
                print(f"{epoch}: Error Avg = {toatalError.item()/len(self.zippedDf):.6f}")


    
    def updateNeuron(self, error, inputs):
        node = self.node
        node.bias = node.bias + node.learningRate * (error)
        for i in range(len(inputs)):
            node.weights[i] = node.weights[i] + ((node.learningRate * (error)) * inputs[i])
