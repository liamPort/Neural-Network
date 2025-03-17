import pandas as pd
import neuralNetwok
import perceptronNN
import adalineNN
import mutliLayerNN as mnn


    


#df = pd.read_csv('out.csv')
#dataSet = list(zip(df[df.columns[~df.columns.isin(['Class'])]].to_numpy(), df[["Class"]].to_numpy()))
df = pd.read_csv('DataSetTwo.csv')
dataSet = list(zip(df[df.columns[~df.columns.isin(['class'])]].to_numpy(), df[["class"]].to_numpy()))
nn = mnn.multiLayerNN(dataSet, 0.1, [4,2,1])
nn.learn(10000)
nn.plotLoss()

