import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import node

class NeuralNetwork:
    def __init__(self, trainingDf, learningRate):
        self.df = trainingDf
        self.zippedDf = list(zip(trainingDf[["x1", "x2"]].to_numpy(), trainingDf[["class"]].to_numpy()))
        self.node = node.Node(learningRate)


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
    
    def printWeightsBias(self):
        print(f"bias: {self.node.bias} + weights: {self.node.weights}")