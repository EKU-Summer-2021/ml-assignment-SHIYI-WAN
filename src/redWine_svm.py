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
    
class RedWineSVM:
    '''
    create a class to initialize svm models
    '''
    def __init__(self):
        '''
        set the configuration of the svm model with a constructor.
        '''
        pd_data = pd.read_csv('winequality-red2.csv')
        data = binarization()
        self.X = pd_data
        self.y = data.iloc[:, -1]
        self.train_X, self.test_X, self.train_y, self.test_y = \
            train_test_split(self.X, self.y, test_size=0.2, random_state=532)

    def build_model(self):
        '''
        training the model with two 
        '''
        model1 = svm.SVC()
        model1.fit(self.train_X, self.train_y)
        prediction = model1.predict(self.test_X)
        print('score', metrics.accuracy_score(prediction, self.test_y))

        model2 = svm.LinearSVC(max_iter=10000)
        model2.fit(self.train_X, self.train_y)
        prediction = model2.predict(self.test_X)
        print('score', metrics.accuracy_score(prediction, self.test_y))
