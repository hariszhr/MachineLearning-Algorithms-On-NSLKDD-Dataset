import random
import math
import numpy as np
import pandas as pd


# Calculate Gaussian Probability
def Gaussian_NB(x, d1, d3):
    exponent = math.exp(-(math.pow(x - d1, 2) / (2 * math.pow(d3, 2))))
    return (1 / (math.sqrt(2 * math.pi) * d3)) * exponent

# Find probability
def class_Prob(meanstd, testdata):
    Gaussian_prob = {}
    for Class_V, classmeanstd in meanstd.items():
        Gaussian_prob[Class_V] = 1
        for i in range(len(classmeanstd)):
            mean, stdev = classmeanstd[i]
            x = testdata[i]
            Gaussian_prob[Class_V] *= Gaussian_NB(x, mean, stdev)
    return Gaussian_prob

# Find prediction
def predict_NB(meanstd, testdata):
    Gaussian_prob = class_Prob(meanstd, testdata)
    label, prob = None, -1
    for Class_V, probability in Gaussian_prob.items():
        if label is None or probability > prob:
            prob = probability
            label = Class_V
    return label

# Get probability
def Probability_NB(meanstd, testdata):
    predictions = []
    for i in range(len(testdata)):
        result = predict_NB(meanstd, testdata[i])
        predictions.append(result)
    return predictions

# Find Accuracy and Confusion matrix
def Accuarcy_confusinmatrix(testdata, predictions):
    correct = 0
    global tp, tn, fp,fn
    tp = 0
    tn=0
    fp=0
    fn=0
    for i in range(len(testdata)):
        if testdata[i][-1] == predictions[i]:
            if(testdata[i][-1] == 0):
                tp += 1
            elif(testdata[i][-1] == 1):
                tn += 1
            correct += 1
        else:
            if predictions[i] == 0:
                fp += 1
            elif predictions[i] == 1:
                fn += 1

    return (correct / float(len(testdata))) * 100.0

def main():
    splitratio = 0.8
    print('Please wait...')
# Reading Data
    _filename = './data/KDDTrain+Top6Attributes.csv'
    df = pd.read_csv(_filename)
    df = df.rename(columns={"class": "label"})
    data = df.values
    header = df.columns
    for idx in range(data.__len__()):
        if data[idx][-1] == "normal":
            data[idx][-1] = 0
        else:
            data[idx][-1] = 1

# Convert service and  flag column into unique numerical values
    abc = data[:, 1]
    abc = list(set(abc))
    for idx in range(data.__len__()):
        data[idx][1] = abc.index(data[idx][1]) + 1
    abc = data[:, 3]
    abc = list(set(abc))
    for idx in range(data.__len__()):
        data[idx][3] = abc.index(data[idx][3]) + 1

# Split the data into training and testing dataset
    random.shuffle(data)
    cut_point = int(len(data) * splitratio)
    traindata = data[:cut_point]
    testdata = data[cut_point:]

# Seperate the label
    splitlabel = {}
    for idx in range(traindata.__len__()):
        attr = traindata[idx]
        if (attr[-1] not in splitlabel):
            splitlabel[attr[-1]] = []
        splitlabel[attr[-1]].append(attr)

# Find mean standard deviation
    meanstd = {}
    for Class_V, value in splitlabel.items():
        #meanstd[Class_V] = combine(value)
        d1 = np.mean(value, axis=0)
        d2 = np.var(value, axis=0)
        att = np.array(value, dtype=np.float)
        d3 = np.std(att, axis=0)
        combine_data = list(zip(d1, d3))
        del combine_data[-1]
        meanstd[Class_V] = combine_data

# Print statement
    predictions = Probability_NB(meanstd, testdata)
    accuracy = Accuarcy_confusinmatrix(testdata, predictions)
    print("Total number of record", len(data))
    print("Split {0} rows into train={1} and test={2} rows".format(len(data), len(traindata), len(testdata)))
    print("\nAccuracy: {0}%".format(accuracy))
    print('\nConfusion Matrix')
    x = np.array([
        ['TN -> ' + str(tn), 'FP -> ' + str(fp)],
        ['FN -> ' + str(fn), 'TP -> ' + str(tp)]
    ], str)
    print(x)
    print('\nSensitivity: ')
    x = tp/(tp+fn)
    print(str(x*100)+'%')
    print('\nSpecificity: ')
    x = tn / (tn + fp)
    print(str(x*100)+'%')

main()