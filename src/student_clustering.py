'''
module of agglomerative clustering
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA


def stdencode():
    '''
    encode
    '''
    pd_data = pd.read_csv('D:\ml-assignment-SHIYI-WAN\StudentsPerformance.csv')
    encoder = preprocessing.OrdinalEncoder()
    encoder.fit(pd_data[['gender']])
    x1 = encoder.transform(pd_data[['gender']])
    d1 = pd.DataFrame(x1, columns=['gender'])
    encoder.fit(pd_data[['race/ethnicity']])
    x2 = encoder.transform(pd_data[['race/ethnicity']])
    d2 = pd.DataFrame(x2, columns=['race/ethnicity'])
    encoder.fit(pd_data[['parental level of education']])
    x3 = encoder.transform(pd_data[['parental level of education']])
    d3 = pd.DataFrame(x3, columns=['parental level of education'])
    encoder.fit(pd_data[['lunch']])
    x4 = encoder.transform(pd_data[['lunch']])
    d4 = pd.DataFrame(x3, columns=['lunch'])
    encoder.fit(pd_data[['test preparation course']])
    x5 = encoder.transform(pd_data[['test preparation course']])
    d5 = pd.DataFrame(x3, columns=['test preparation course'])
    d = pd.concat([d1, d2, d3, d4, d5], axis=1)
    pd_data.drop(columns=['gender', 'race/ethnicity',
                          'parental level of education',
                          'lunch', 'test preparation course'], inplace=True)
    data = pd.concat([pd_data, d], axis=1)
    return data


def scaler():
    '''
    normalize
    '''
    data = stdencode()
    min_max_scaler = preprocessing.MinMaxScaler()
    data_s = min_max_scaler.fit_transform(data)
    return data_s


class Agglomerative:
    '''
    class to init agglomeraative cluster
    '''
    def __init__(self):
        self.data = scaler()
        self.labels = 0

    def training_model(self):
        '''
        train data
        '''
        ac = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
        ac.fit(self.data)
        self.labels = ac.fit_predict(self.data)
        print(self.labels)

    def visual_pca(self):
        '''
        visualize
        '''
        pca = PCA(n_components=2)
        reduced_data_pca = pca.fit_transform(self.data)
        print(reduced_data_pca)
        x = reduced_data_pca[:, 0]
        y = reduced_data_pca[:, 1]
        plt.scatter(x, y, c=self.labels)
        plt.show()
