import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import node


def showDataPlot(df, node: node.Node):
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
    


df = pd.read_csv('DataSetOne.csv')

node_one = node.Node()

node_one.learn(df.iloc[[0], [0,1]].to_numpy()[0], df.iloc[[0], [2]].to_numpy()[0])


zippedData = list(zip(df[["x1", "x2"]].to_numpy(), df[["class"]].to_numpy()))

t = False
count = 0
while t != True:
    count += 1
    t = True
    for inputs, target in zippedData:
        result = node_one.learn(inputs, target)
        if result == False:
            t = False


print(count)
showDataPlot(df, node_one)