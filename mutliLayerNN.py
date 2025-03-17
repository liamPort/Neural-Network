import pandas as pd
import matplotlib.pyplot as plt
import utils
import node
import numpy as np

class multiLayerNN():
    def __init__(self, trainingDataSet, learningRate, layerNums):
        self.DataSet = trainingDataSet
        self.learningRate = learningRate
        self.nodeLayers = []
        self.totalMse = []
        self.createNodes(layerNums)
    
    def createNodes(self, layerNums):
        for layerIndex in range(len(layerNums)):
            self.nodeLayers.append([])
            if(layerIndex == 0):
                inputSize = len(self.DataSet[0][0])
            else:
                inputSize = layerNums[layerIndex - 1]
            for nodeIndex in range(layerNums[layerIndex]):
                self.nodeLayers[layerIndex].append(node.Node(inputSize))
    
    def forwardPass(self, inputs):
        activations = []
        weightedSums = []
        for layerIndex in range(len(self.nodeLayers)):
            #passes through each layer
            activations.append([])
            weightedSums.append([])
            for nodeIndex in range(len(self.nodeLayers[layerIndex])):
                node = self.nodeLayers[layerIndex][nodeIndex]
                if layerIndex == 0:
                    weightedSum = self.predict(node, inputs)
                    activations[layerIndex].append(weightedSum)
                elif layerIndex == len(self.nodeLayers) - 1:
                    #if at output node
                    weightedSum = self.predict(node, activations[layerIndex-1])
                    activations[layerIndex].append(weightedSum)
                else:
                    #if its a inner layer, pridict based on previcious layer
                    weightedSum = self.predict(node, activations[layerIndex-1])
                    activations[layerIndex].append(utils.sigmoid(weightedSum))
                weightedSums[layerIndex].append(weightedSum)
        return activations, weightedSums, activations[layerIndex][nodeIndex]


    def backwardPass(self, layerActivations, layerweightedSums, target, inputs):
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
                if(lastNodes):
                    LayerErrorTerms[layerIndex][nodeIndex] = ((target - layerActivations[layerIndex][nodeIndex]))[0]
                else:
                    #calculate weightedError for each node infront
                    weightedSumError = 0
                    for forwardNodeIndex in range(len(self.nodeLayers[layerIndex + 1])):
                        fwNode: node.Node = self.nodeLayers[layerIndex + 1][forwardNodeIndex]
                        weightedSumError += fwNode.weights[nodeIndex] * LayerErrorTerms[layerIndex + 1][forwardNodeIndex]
                    LayerErrorTerms[layerIndex][nodeIndex] = weightedSumError * activationGradient
                currentNode = self.nodeLayers[layerIndex][nodeIndex]
                if layerIndex > 0:
                    self.updateNode(currentNode, LayerErrorTerms[layerIndex][nodeIndex], layerActivations[layerIndex - 1])
                else:
                    self.updateNode(currentNode, LayerErrorTerms[layerIndex][nodeIndex], inputs)


    def updateNode(self, node: node.Node, errorTerm, previousLayer):
        node.bias += (self.learningRate * errorTerm)
        for weightIndex in range(len(node.weights)):
            originalWeight = node.weights[weightIndex]
            node.weights[weightIndex] += (self.learningRate * errorTerm * previousLayer[weightIndex]) + (0.9 * node.velocities[weightIndex])
            node.velocities[weightIndex] = node.weights[weightIndex] - originalWeight


    def plotLoss(self):
        plt.plot(self.totalMse)
        plt.show()
        
        
    
    def predict(self, node: node.Node, inputs):
        #print(f"bias= {node.bias}")
        weightedSum = float(node.bias)
        for i in range(len(inputs)):
            weightedSum += node.weights[i] * inputs[i]
        return weightedSum
    
    def learn(self, epochs):
        for epoch in range(epochs):
            predictions = []
            targets = []
            for inputs, target in self.DataSet:
                layerActivations, layerweightedSums, prediction = self.forwardPass(inputs)
                predictions.append(prediction)
                targets.append(target)
                self.backwardPass(layerActivations, layerweightedSums, target, inputs)
            

            self.totalMse.append(utils.mseCalculation(predictions, targets))     

                
        




