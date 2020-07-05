import operator
import random
import math
import pandas as pd
import numpy as np


def loadDataset(filename, fsplit, trainingSet=[], testSet=[]):

    print('Please wait...')
    # Reading Data
    df = pd.read_csv(filename)
    df = df.rename(columns={"class": "label"})
    data = df.values
    for idx in range(data.__len__()):
        if data[idx][-1] != "normal":
            data[idx][-1] = "anomaly"
    # Convert service and  flag column into unique numerical values
    abc = data[:, 1]
    abc = list(set(abc))
    for idx in range(data.__len__()):
        data[idx][1] = abc.index(data[idx][1]) + 1
    abc = data[:, 3]
    abc = list(set(abc))
    for idx in range(data.__len__()):
        data[idx][3] = abc.index(data[idx][3]) + 1

    for x in range(len(data) - 1):
        if random.random() >= fsplit:
            testSet.append(data[x])
        else:
            trainingSet.append(data[x])


def getEuclideanDistance(a, b, size):
    distance = 0
    for x in range(size):
        num1=float(a[x])
        num2=float(b[x])
        distance += pow((num1- num2), 2)
    return math.sqrt(distance)

def getNeighbors(trainingSet, testInstance, k):
   distances = []
   length = len(testInstance)-1
   for x in range(len(trainingSet)):
        dist = getEuclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
   distances.sort(key=operator.itemgetter(1))
   neighbors = []
   for x in range(k):
        neighbors.append(distances[x][0])
   return neighbors

def getPrediction(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]


def main():
    #prepare data
    trainingSet=[]
    testSet=[]
    split =0.80
    _filename = './data/20PercentDataWithTop6Attributes.csv'
    loadDataset(_filename, split,trainingSet, testSet)
    print ('Train set: '+repr(len(trainingSet)))
    print ('test set: '+repr(len(testSet)))
    #generate predictions
    presictions=[]
    k=3

    print('kNN algorithm is running. This may take some time. please wait...')

    l = 0
    while (l < len(testSet)):
       neighbours = getNeighbors(trainingSet, testSet[l], k)
       result = getPrediction(neighbours)
       presictions.append(result)
       # print(presictions[l])
       # print('> predicted =' + repr(result) + ', actual=' + repr(testSet[l][-1]))
       l += 1

    global tp, tn, fp, fn
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] in presictions[x]:
            if testSet[x][-1] == "normal":
                tp += 1
            elif testSet[x][-1] == "anomaly":
                tn += 1
            correct += 1
        else:
            if presictions[x] == "normal":
                fp += 1
            elif presictions[x] == "anomaly":
                fn += 1
    accuracy = (correct / float(len(testSet))) * 100.0

    print("\nAccuracy: {0}%".format(accuracy))
    print('\nConfusion Matrix')
    x = np.array([
        ['TN -> ' + str(tn), 'FP -> ' + str(fp)],
        ['FN -> ' + str(fn), 'TP -> ' + str(tp)]
    ], str)
    print(x)
    print('\nSensitivity: ')
    tpfn = tp + fn
    if tpfn > 0:
        x = tp / (tp + fn)
    else:
        x = 1.0
    print(str(x*100)+'%')
    print('\nSpecificity: ')
    tnfp = tn + fp
    if tnfp > 0:
        x = tn / (tn + fp)
    else:
        x = 1.0
    print(str(x*100)+'%')

main()


##referece:
##https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/