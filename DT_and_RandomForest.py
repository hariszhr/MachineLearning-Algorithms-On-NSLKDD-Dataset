# from __future__ import division # to be removed for python 3.x
import subprocess
import sys

def install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])


install('numpy')
install('pandas')


import pandas as pd
import numpy as np
from pprint import pprint
import random as random
random.seed(9001)
np.random.seed(9001)
# pprint = pprint.PrettyPrinter(indent=4)


_trainDataSplitRatio = 0.2
_randomForestIterations = 10
_randomForestModels = []
_fullDecisionTreeModel = ""


_filename = './data/KDDTrain+Top6Attributes.csv'
# _filename = './data/20PercentDataWithTopAttrs.csv'
_df = pd.read_csv(_filename)
_df = _df.rename(columns={"class": "label"})
# _df

_data = _df.values
_headers = list(_df.columns)
_features = range(_data.shape[1] - 1)



# Step3 convert all non-normal labels to anomaly
print("step: converting all anomaly class labels to 'anomaly'".upper())
for idx in range(_data.__len__()):
    if _data[idx][-1:] != 'normal':
        _data[idx][-1:] = 'anomaly'
# _data
print('done'.upper())

def convert_cont_data_to_discrete(data, column_idx, headers):
    print(str('step: converting continuous data into discrete data').upper() + ' - column [' + headers[column_idx] + ']\nPlease wait...')

    class_labels = data[:, -1]

    attribute_col_data = data[:, column_idx]

    attribute_col_unique_values = list(set(attribute_col_data))

    attribute_col_unique_values.sort()

    #     print('attribute_col_unique_values', attribute_col_unique_values)

    weighted_gini_impurities_for_each_split = []
    data_for_weighted_gini_impurities_for_each_split = []
    range_val_list = []

    #   Pick value b/w each unique value of this attr and find gini_impurity for that range
    for idx in range(attribute_col_unique_values.__len__() - 1):

        idx2 = idx * 10

        if idx2 >= (attribute_col_unique_values.__len__() - 1):
            break

        tmpArr = []
        for e1 in attribute_col_data:
            tmpArr.append(e1)

        range_val = ((float)(attribute_col_unique_values[idx2]) + (float)(attribute_col_unique_values[idx2 + 1])) / 2.0
        range_val_array = ['<' + str(range_val), '>' + str(range_val)]

        #         print('***', range_val_array)
        #         print(tmpArr)
        for ii in range(tmpArr.__len__()):
            if (tmpArr[ii] > range_val):
                tmpArr[ii] = range_val_array[1]
            elif (tmpArr[ii] < range_val):
                tmpArr[ii] = range_val_array[0]
            else:
                print ('ERROR')

        #         print(tmpArr)

        gini_impurity_lists = {}
        attr_discrete_value_count = {}
        total_count = (float)(data.__len__())
        normal_count = 0.0
        anomaly_count = 0.0

        for each in range_val_array:
            normal_count = 0.0
            anomaly_count = 0.0
            for idx3 in range(tmpArr.__len__()):
                if tmpArr[idx3] == each:
                    if class_labels[idx3] == "normal":
                        normal_count += 1.0
                    elif class_labels[idx3] == "anomaly":
                        anomaly_count += 1.0
                    else:
                        print("ERROR -- class_labels[idx3] =>", class_labels[idx3])

            valuee = 1.0 - ((float(normal_count) / float(normal_count + anomaly_count) * float(normal_count) / float(
                normal_count + anomaly_count))
                            + (float(anomaly_count) / float(normal_count + anomaly_count) * float(
                        anomaly_count) / float(normal_count + anomaly_count)))
            #             print("---> ", valuee)
            gini_impurity_lists[each] = valuee
            attr_discrete_value_count[each] = float(normal_count + anomaly_count)
        #             total_count += float(normal_count+anomaly_count)

        #       Now calc weighted gini impurity for all values of this attribute

        weighted_gini_impurity = 0.0
        for i in gini_impurity_lists:
            weighted_gini_impurity += (attr_discrete_value_count[i] / total_count) * gini_impurity_lists[i]

        weighted_gini_impurities_for_each_split.append(weighted_gini_impurity)
        data_for_weighted_gini_impurities_for_each_split.append(tmpArr)
        range_val_list.append(range_val)

    list1, list3 = zip(*sorted(zip(weighted_gini_impurities_for_each_split, range_val_list)))

    print('[BEST GINI_IMPURITY] ', list1[0])
    print('[BEST SPLIT POINT] ', list3[0])
    #     print('[SPLIT DATA] ', list2[0])

    split_val = list3[0]
    arr = data[:, column_idx]
    split_val_array = ['<' + str(split_val), '>' + str(split_val)]
    for ii in range(arr.__len__()):
        if arr[ii] > split_val:
            arr[ii] = split_val_array[1]
        elif arr[ii] < split_val:
            arr[ii] = split_val_array[0]
        else:
            print ('ERROR')
    print('-----------------------------------------------------------------------------------\n')


convert_cont_data_to_discrete(_data, 0, headers=_headers)
convert_cont_data_to_discrete(_data, 2, headers=_headers)
convert_cont_data_to_discrete(_data, 4, headers=_headers)
convert_cont_data_to_discrete(_data, 5, headers=_headers)


# SPLIT TEST AND TRAIN DATA
tdSize = int(len(_data) * (1-_trainDataSplitRatio))
testdSize = len(_data) - tdSize
# print(tdSize, testdSize)
_testData = _data[-testdSize:, :]
_data = _data[:tdSize, :]
print('train data size', _data.__len__())
print('test data size', _testData.__len__())
print('\n')


def gini_impurity(data, column_idx):
#     print('\tfinding gini impurity for column', column_idx)
    class_col_data = data[:, -1] # class labels
    attribute_col_data = data[:, column_idx] # attribute data
    gini_impurity_lists = {}
    attr_discrete_value_count = {}
    total_count = 0
    for each in np.unique(attribute_col_data):
        normal_count = 0
        anomaly_count = 0
        for idx in range(attribute_col_data.__len__()):
                if attribute_col_data[idx] == each:
                    if class_col_data[idx] == "normal":
                        normal_count += 1
                        total_count += 1
                    elif class_col_data[idx] == "anomaly":
                        anomaly_count += 1
                        total_count += 1
                    else:
                        print("ERROR - class value is neither normal nor anomaly")
        if normal_count == 0 or anomaly_count == 0:
            gini_impurity_lists[each] = 0.0
        else:
            value = 1.0 - ((float(normal_count)/float(normal_count+anomaly_count) * float(normal_count)/float(normal_count+anomaly_count))
                            +(float(anomaly_count)/float(normal_count+anomaly_count) * float(anomaly_count)/float(normal_count+anomaly_count)))
            gini_impurity_lists[each] = value
        attr_discrete_value_count[each] = float(normal_count+anomaly_count)

    weighted_gini_impurity=0.0
    for i in gini_impurity_lists:
        weighted_gini_impurity += (attr_discrete_value_count[i]/float(total_count)) * gini_impurity_lists[i]

#     print('gini imputiry', weighted_gini_impurity)
    return weighted_gini_impurity


def find_col_with_min_gini_impurity(data):
    gini_impurities={}

    features = range(data.shape[1] - 1)
#     print('features', features)

    for column_idx in features:
        gini_impurities[column_idx]=gini_impurity(data, column_idx)
    # find minimum gini impurity
    mini=-1
    column=-1
    for k in gini_impurities:
        if mini == -1:
            mini = gini_impurities.get(k)
            column=k
        if gini_impurities.get(k) < mini:
            mini = gini_impurities.get(k)
            column = k
#     print('column with minimum gini impurity for given data', column, 'gini value', mini)
    return column



def is_pure(data):
    unique_values = np.unique(data[:, -1])
    if unique_values.__len__() == 1:
        return True
    else:
        return False


def leaf_data(data):
    unique_values, counts = np.unique(data[:, -1], return_counts=True)
    max_index = counts.argmax()
    return unique_values[max_index]


def split(data, column_index):
    splits = []
    split_col_values = data[:, column_index]
    for each in np.unique(split_col_values):
        splits.append(data[split_col_values == each])
    return splits


_df = pd.DataFrame(data=_data, columns=_headers)
# _df


def _algorithm(data, depth=0, counterLimit=6):
    if is_pure(data) or depth == counterLimit:
        #         if depth==counterLimit:
        #             print('ALERT! counter limit is hit', depth)
        #         print('CLASSIFYING DATA')
        return leaf_data(data)

    else:
        split_column_index = find_col_with_min_gini_impurity(data)
        split_data_list = split(data, split_column_index)

        depth += 1
        #         print('depth',depth)

        node = _headers[split_column_index]

        jsonTree = {node: []}

        for each in split_data_list:
            #             print('split_data_list',each)
            value = _algorithm(each, depth)
            branch_name = str(np.unique(each[:, split_column_index])[0])
            jo = {branch_name: value}

            #             print('new node',jo)

            jsonTree[node].append(jo)

        return jsonTree


_fullDecisionTreeModel = _algorithm(_df.values, counterLimit=_features)

print(' ########### FULL DECISION TREE MODEL ########### ')
pprint(_fullDecisionTreeModel)
print(' ########### END - FULL DECISION TREE MODEL ########### ')

rows_in_each_iter = int(_data.__len__()/2)
for i in range(_randomForestIterations):
    rows = np.random.randint(_data.__len__(), size=rows_in_each_iter)
    mydata = _data[rows,:]
    _randomForestModels.append(_algorithm(mydata, counterLimit=_features))

print(' ########### RANDOM FOREST MODELS ########### ')
print('# of models: ' + str(len(_randomForestModels)))
for i in range(len(_randomForestModels)):
    print('MODEL', i+1)
    pprint(_randomForestModels[i])
    print('\n\n')
print(' ########### END - RANDOM FOREST MODELS ########### ')


def readTreeModel(model, row_dict,  headers=_headers):
    # pprint(_fullDecisionTreeModel)
    # if isinstance(ele,dict):
    for k, v in model.items():
        if isinstance(v, str):
            rslt = row_dict['label'] == v
            if rslt:
                if v == 'normal':
                    rslt = 'TP'
                elif v == 'anomaly':
                    rslt = 'TN'
                else:
                    print('ERROR rslt', rslt)
            else:
                if v == 'normal':
                    rslt = 'FP'
                elif v == 'anomaly':
                    rslt = 'FN'
                else:
                    print('ERROR rslt', rslt)
            return rslt
        elif isinstance(v, dict):
            return readTreeModel(v, row_dict)
        elif isinstance(v, list):
            # print('list found', v.__len__())
            tmp = row_dict[k]
            for each in v:
                if tmp in each.keys():
                    # print('matched '+ k+" --> "+ tmp)
                    return readTreeModel(each, row_dict)
        else:
            print('else', k, v)


def most_frequent(List):
    return max(set(List), key = List.count)


def predictClassifier(testData, model):
    if isinstance(model, dict):
        model = [model]

    rs_list = []
    for row in _testData:
        # print('\n*** ROW' + str(row))
        myrow_dict = {}
        for each in _headers:
            myrow_dict[each] = row[_headers.index(each)]
        results = []
        for i in range(len(model)):
            # print('TREE: ' + str(i))
            results.append(readTreeModel(model[i], myrow_dict))
        result = most_frequent(results)
        rs_list.append(result)


    rs_list2=[]
    for item in rs_list:
        if item is not None:
            rs_list2.append(item)

    a, b = np.unique(np.array(rs_list2), return_counts=True)

    if 'TP' in a:
        tp = b[list(a).index('TP')]
    else:
        tp = 0
    if 'TN' in a:
        tn = b[list(a).index('TN')]
    else:
        tn = 0
    if 'FP' in a:
        fp = b[list(a).index('FP')]
    else:
        fp = 0
    if 'FN' in a:
        fn = b[list(a).index('FN')]
    else:
        fn = 0
    # tn = b[list(a).index('TN')]
    # if tp is None:
    #     tp=0
    # if tn is None:
    #     tn = 0
    acc = (float(tp + tn) / len(_testData)) * 100
    print('ACCURACY: ' + str(acc) + '%')
    print('\nCONFUSION Matrix')
    x = np.array([
        ['TN -> ' + str(tn), 'FP -> ' + str(fp)],
        ['FN -> ' + str(fn), 'TP -> ' + str(tp)]
                ], str)
    print(x)
    print('Sensitivity: ')
    x = float(tp) / float((tp + fn))
    print(str(x * 100) + '%')
    print('Specificity: ')
    x = float(tn) / float((tn + fp))
    print(str(x * 100) + '%')

print('\nDecision Tree Stats:')
predictClassifier(_testData, _fullDecisionTreeModel)

print('\nRandom Forest Tree Stats:')
predictClassifier(_testData, _randomForestModels)


