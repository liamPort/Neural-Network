import pandas as pd
import neuralNetwok
import perceptronNN
import adalineNN


    


df = pd.read_csv('DataSetOne.csv')

nn = adalineNN.adalineNN(df, 0.02)
nn.learn(1000)
nn.printWeightsBias()



nn.showDataPlot()