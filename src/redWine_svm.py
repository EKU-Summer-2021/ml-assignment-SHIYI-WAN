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
