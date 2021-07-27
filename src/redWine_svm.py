import csv
import pandas as pd
from sklearn import preprocessing, svm, metrics
from sklearn.model_selection import train_test_split


def binarization():
    '''
    binarize the data
    '''
    data = pd.read_csv('winequality-red.csv')
    binarizer = preprocessing.Binarizer(threshold=data['quality'].mean())
    y = binarizer.transform(data[['quality']])
    data['quality'] = y
    return data

def normalize():
    '''
    normalization data
    '''
    data =pd.read_csv('winequality-red.csv')
    sam = []
    a = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
         'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
         'pH', 'sulphates', 'alcohol']
    for i in a:
        y = data.loc[:, i]
        ys = list(preprocessing.scale(y))
        sam.append(ys)

    print(len(sam))
    with open('winequality-red2.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerow(['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                         'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                         'pH', 'sulphates', 'alcohol'])
        for i in range(len(sam[0])):
            writer.writerow(
                [sam[0][i], sam[1][i], sam[2][i], sam[3][i], sam[4][i], sam[5][i], sam[6][i], sam[7][i], sam[8][i],
                 sam[9][i], sam[10][i]])
    print("done")
