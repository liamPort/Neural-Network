import pandas as pd
import neuralNetwok

class perceptronNN(neuralNetwok.NeuralNetwork):
    def __init__(self, trainingDf, learningRate):
        super().__init__(trainingDf, learningRate)

    
    def predict(self, inputs):
        node = self.node
        weightedSum = float(node.bias)
        for i in range(len(inputs)):
            weightedSum += node.weights[i] * inputs[i]
        return self.activation(weightedSum)
    
    def activation(x):
        if x >=0:
            return 1
        else:
            return -1
    
    def learn(self):
        t = False
        while t != True:
            t = True
            for inputs, target in self.zippedDf:
                result = self.trainPerceptron(inputs, target)
                if result == False:
                    t = False
    
    def trainPerceptron(self, inputs, target):
        node = self.node
        prediction = node.predict(inputs)
        if(prediction != target):
            node.bias = node.bias + (target * node.learningRate)
            for i in range(len(inputs)):
                node.weights[i] = node.weights[i] + ((target * node.learningRate) * inputs[i])
            return False
        return True