import pandas as pd
import neuralNetwok


    


df = pd.read_csv('DataSetOne.csv')

nn = neuralNetwok.NeuralNetwork(df)
nn.learn(nn.perceptronLearning)



nn.showDataPlot()