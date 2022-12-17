from NeuralNetUtil import buildExamplesFromCarData,buildExamplesFromPenData
from NeuralNet import buildNeuralNet
from math import pow, sqrt
import statistics

def average(argList):
    return sum(argList)/float(len(argList))

def stDeviation(argList):
    mean = average(argList)
    diffSq = [pow((val-mean),2) for val in argList]
    return sqrt(sum(diffSq)/len(argList))

penData = buildExamplesFromPenData()
def testPenData(hiddenLayers = [24]):
    return buildNeuralNet(penData, maxItr = 200, hiddenLayerList = hiddenLayers)

carData = buildExamplesFromCarData()
def testCarData(hiddenLayers = [16]):
    return buildNeuralNet(carData, maxItr = 200,hiddenLayerList = hiddenLayers)

for hl in range(0, 41, 5):
    runlist = []
    for _ in range(5):
        nn, acc = testPenData(hiddenLayers=[hl])
        runlist.append(acc)
    print(f'hiddenLayers:{hl}, maxacc:{max(runlist)}, avgacc:{statistics.mean(runlist)}, stddev:{statistics.stdev(runlist)}')
