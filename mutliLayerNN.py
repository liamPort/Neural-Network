import pandas as pd
import neuralNetwok
import random
import utils
import node
import numpy as np

class multiLayerNN():
    def __init__(self, trainingDataSet, learningRate, Layers):
        self.DataSet = trainingDataSet
        self.learningRate = learningRate
        self.nodeLayers = []

        #create the node layers 
        for layerIndex in range(len(Layers)):
            self.nodeLayers.append([])
            if(layerIndex == 0):
                inputSize = len(trainingDataSet[0][0])
            else:
                inputSize = Layers[layerIndex - 1]
            for nodeIndex in range(Layers[layerIndex]):
                self.nodeLayers[layerIndex].append(node.Node(inputSize))
                #self.nodeLayers[layerIndex][nodeIndex].printNodeInfo(f"[{layerIndex}][{nodeIndex}]")
    
    def forwardPass(self, inputs):
        activations = []
        weightedSums = []
        for layerIndex in range(len(self.nodeLayers)):
            activations.append([])
            weightedSums.append([])
            for nodeIndex in range(len(self.nodeLayers[layerIndex])):
                node = self.nodeLayers[layerIndex][nodeIndex]
                if layerIndex == 0:
                    weightedSum = self.predict(node, inputs)
                else:
                    weightedSum = self.predict(node, activations[layerIndex-1])
                activations[layerIndex].append(utils.sigmoid(weightedSum))
                weightedSums[layerIndex].append(weightedSum)
        return activations, weightedSums, activations[layerIndex][nodeIndex]
    
    def backwardPass(self, layerActivations, layerweightedSums, target):
        LayerErrorTerms =  []
        #set LayerErrorTerms to have the same size as layerActivations
        for layer in layerActivations:
            LayerErrorTerms.append([0] * len(layer))
        
        #go backwards in the layer arrays
        for layerIndex in reversed(range(len(self.nodeLayers))):
            if layerIndex == (len(self.nodeLayers) -1):
                lastNodes = True
            else:
                lastNodes = False
            
            for nodeIndex in range(len(self.nodeLayers[layerIndex])):
                activationGradient = utils.derivativeSigmoid(layerweightedSums[layerIndex][nodeIndex])
                activatedSum = layerActivations[layerIndex][nodeIndex]
                if(lastNodes):
                    LayerErrorTerms[layerIndex][nodeIndex] = ((target - activatedSum) * activationGradient)[0]
                else:
                    #calculate weightedError for each node infront
                    weightedSumError = 0
                    for forwardNodeIndex in range(len(self.nodeLayers[layerIndex + 1])):
                        fwNode: node.Node = self.nodeLayers[layerIndex + 1][forwardNodeIndex]
                        weightedSumError += fwNode.weights[nodeIndex] * LayerErrorTerms[layerIndex + 1][forwardNodeIndex]
                    LayerErrorTerms[layerIndex][nodeIndex] = weightedSumError * activationGradient
                currentNode = self.nodeLayers[layerIndex][nodeIndex]
                self.updateNode(currentNode, LayerErrorTerms[layerIndex][nodeIndex], activatedSum)


    def updateNode(self, node: node.Node, errorTerm, activatedSum):
        node.bias = node.bias + (self.learningRate * errorTerm * activatedSum)
        for weightIndex in range(len(node.weights)):
            node.weights[weightIndex] = node.weights[weightIndex] + (self.learningRate * errorTerm * activatedSum)


        
        
    
    def predict(self, node: node.Node, inputs):
        #print(f"bias= {node.bias}")
        weightedSum = float(node.bias)
        for i in range(len(inputs)):
            weightedSum += node.weights[i] * inputs[i]
        return weightedSum
    
    def learn(self, epochs):
        for epoch in range(epochs):
            for inputs, target in self.DataSet:
                layerActivations, layerweightedSums, prediction = self.forwardPass(inputs)
                self.backwardPass(layerActivations, layerweightedSums, target)
            
                
        


df = pd.read_csv('DataSetOne.csv')
dataSet = list(zip(df[["x1", "x2"]].to_numpy(), df[["class"]].to_numpy()))
nn = multiLayerNN(dataSet, 0.001, [2, 1])
nn.learn(10000)

