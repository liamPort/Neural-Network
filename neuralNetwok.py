import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import node

class NeuralNetwork:
    def __init__(self, trainingDf):
        self.df = trainingDf
        self.zippedDf = list(zip(trainingDf[["x1", "x2"]].to_numpy(), trainingDf[["class"]].to_numpy()))
        self.node = node.Node()

    def learn(slef, learningFunction):
        learningFunction()
    
    def perceptronLearning(self):
        t = False
        while t != True:
            t = True
            for inputs, target in self.zippedDf:
                result = self.trainPerceptron(inputs, target)
                if result == False:
                    t = False
    
    def trainPerceptron(self, inputs, target):
        node = self.node
        prediction = self.node.predict(inputs)
        if(prediction != target):
            node.bias = node.bias + (target * 0.1)
            for i in range(len(inputs)):
                node.weights[i] = node.weights[i] + ((target * 0.1) * inputs[i])
            return False
        return True


    def showDataPlot(self):
        df = self.df
        node = self.node

        #plots the datapoints from the dataframe
        df["color"] = np.where(df["class"]==1, "blue", "red")
        df.plot(kind = 'scatter', x = 'x1', y = 'x2', color=df["color"])

        w1, w2 = node.getWeights()
        b = node.getBias()

        #chat GPT underneath
        x_vals = np.linspace(df["x1"].min(), df["x1"].max(), 100)
        if w2 != 0:  # Avoid division by zero
            y_vals = - (w1 / w2) * x_vals - (b / w2)
            plt.plot(x_vals, y_vals, color='black', linestyle='--', label="Decision Boundary")
        
        plt.show()