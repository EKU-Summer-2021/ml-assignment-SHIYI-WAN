'''
module for SVM
'''
import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing, svm, metrics
from sklearn.model_selection import train_test_split


def binarization():
    '''
    binarize the data
    '''
    data = pd.read_csv('winequality-red.csv')
    binarizer = preprocessing.Binarizer(threshold=data['quality'].mean())
    encoder = binarizer.transform(data[['quality']])
    data['quality'] = encoder
    return data


def normalize():
    '''
    normalization data
    '''
    data = pd.read_csv('winequality-red.csv')
    sam = []
    array = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
             'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
             'pH', 'sulphates', 'alcohol']
    for i in array:
        elm = data.loc[:, i]
        elms = list(preprocessing.scale(elm))
        sam.append(elms)

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
    rea wine class to init svm model
    '''

    def __init__(self):
        pd_data = pd.read_csv('winequality-red2.csv')
        data = binarization()
        self.pd_x = pd_data
        self.pd_y = data.iloc[:, -1]
        self.train_x, self.test_x, self.train_y, self.test_y = \
            train_test_split(self.pd_x, self.pd_y, test_size=0.2, random_state=532)
        self.score1 = 0
        self.w = 0
        self.b = 0
        self.n_Support_vector = 0
        self.sv_idx = 0
    def build_model(self):
        '''
        train the model with data
        '''
        model1 = svm.SVC()
        model1.fit(self.train_x, self.train_y)
        prediction = model1.predict(self.test_x)
        self.score1 = metrics.accuracy_score(prediction, self.test_y)
        print('score', self.score1)
        model2 = svm.SVC(kernel='linear', max_iter=10000)
        model2.fit(self.train_x, self.train_y)
        prediction = model2.predict(self.test_x)
        self.w = model2.coef_
        self.b = model2.intercept_
        self.n_Support_vector = model2.n_support_
        self.sv_idx = model2.support_
        print('score', metrics.accuracy_score(prediction, self.test_y))

    def vistualize(self):
        ax = plt.subplot(111, projection='3d')
        x = np.arange(0, 1, 0.01)
        y = np.arange(0, 1, 0.11)
        x, y = np.meshgrid(x, y)
        z = (self.w[0, 0] * x + self.w[0, 1] * y + self.b) / (-self.w[0, 2])
        surf = ax.plot_surface(x, y, z, rstride=1, cstride=1)
        x_array = np.array(self.train_x, dtype=float)
        y_array = np.array(self.train_y, dtype=int)
        pos = x_array[np.where(y_array == 1)]
        neg = x_array[np.where(y_array == -1)]
        ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c='r', label='pos')
        ax.scatter(neg[:, 0], neg[:, 1], neg[:, 2], c='b', label='neg')
        X = np.array(self.train_x, dtype=float)
        for i in range(len(self.sv_idx)):
            ax.scatter(X[self.sv_idx[i], 0], X[self.sv_idx[i], 1], X[self.sv_idx[i], 2], s=50,
                       marker='o', edgecolors='g')

        ax.set_zlabel('Z')
        ax.set_ylabel('Y')
        ax.set_xlabel('X')
        ax.set_zlim([0, 1])
        plt.legend(loc='upper left')

        ax.view_init(35, 300)
        plt.show()
