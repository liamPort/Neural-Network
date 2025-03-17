import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def addClassColumn(df):
    df['Class'] = df['Skelton'].shift(-1)
    df = df[:-1]
    return df

def changeDate(df):
    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
    df['date'] = df['date'].apply(calculateNormalisedDate)
    return df

def cleanDF(df):
     df = df.apply(pd.to_numeric, errors='coerce')
     df[df > 200] = np.nan
     df[df < 0] = np.nan
     df = df.dropna()
     return df

def calculateNormalisedDate(data):
    dayNum = data.dayofyear
    #this makes it so (1, 12) mounths are 0 and (6) is 1
    if dayNum <= 183:
        return round((dayNum - 1) / 183, 4)
    else:
        return round(1 - ((dayNum - 184) / 183), 4)
    
def showDataPlot(df):
        #plots the datapoints from the dataframe
        df['Class'] = pd.to_numeric(df['Class'], errors='coerce')
        df['Snaizeholme'] = pd.to_numeric(df['Snaizeholme'], errors='coerce')
        df.plot(kind = 'scatter', x = 'Snaizeholme', y = 'Class')
        plt.show()



df = pd.read_csv('Data/originalData.csv')
df = addClassColumn(df)
df = changeDate(df)
df = cleanDF(df)

print(df)
showDataPlot(df)
df.to_csv('out.csv', index=False)
