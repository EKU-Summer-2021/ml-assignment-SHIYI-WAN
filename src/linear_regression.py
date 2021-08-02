'''
module for linear regression
'''
import csv
import urllib.request
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


def load_data(path, filename):
    '''
    function load data from url
    '''
    urllib.request.urlretrieve(path, filename)
    data = pd.read_csv(filename)
    return data


def display_lr():
    '''
    function to display the performance
    '''
    pd_data = pd.read_csv('insurance2.csv')
    print('pd_data.head(10)=\n{}'.format(pd_data.head(10)))
    plt.rcParams['axes.unicode_minus'] = False
    sns.pairplot(pd_data, x_vars=['age', 'bmi', 'children', 'region', 'sex', 'smoker'], y_vars='charges', kind="reg",
                 height=5, aspect=0.7)
    plt.show()


def encode():
    '''
    encode data
    '''
    pd_data = pd.read_csv('D:\ml-assignment-SHIYI-WAN\insurance.csv')
    encoder = preprocessing.OrdinalEncoder()
    encoder.fit(pd_data[['region']])
    r_data = encoder.transform(pd_data[['region']])
    r_pd = pd.DataFrame(r_data, columns=['region'])
    encoder.fit(pd_data[['sex']])
    s_data = encoder.transform(pd_data[['sex']])
    s_pd = pd.DataFrame(s_data, columns=['sex'])
    encoder.fit(pd_data[['smoker']])
    sm_data = encoder.transform(pd_data[['smoker']])
    sm_pd = pd.DataFrame(sm_data, columns=['smoker'])
    all_data = pd.concat([r_pd, s_pd, sm_pd], axis=1)
    pd_data.drop(columns=['sex', 'smoker', 'region'], inplace=True)
    data = pd.concat([pd_data, all_data], axis=1)
    return data


def normalization():
    '''
    normalization data
    '''
    data = encode()
    sam = []
    array = ['age', 'bmi', 'children', 'charges', 'region', 'sex', 'smoker']
    for i in array:
        elm = data.loc[:, i]
        elms = list(preprocessing.scale(elm))
        sam.append(elms)

    print(len(sam))
    with open('insurance2.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerow(['age', 'bmi', 'children', 'charges', 'region', 'sex', 'smoker'])
        for i in range(len(sam[0])):
            writer.writerow([sam[0][i], sam[1][i], sam[2][i], sam[3][i], sam[4][i], sam[5][i], sam[6][i]])
    print("done")


class LinearRegressor:
    '''
    create a class to initialize Linear Regression models
    '''

    def __init__(self):
        '''
        set the configuration of the Linear Regression model with a constructor.
        '''
        pd_data = pd.read_csv('insurance2.csv')
        self.pd_x = pd_data.loc[:, ('age', 'sex', 'bmi', 'children', 'region', 'smoker')]
        self.pd_y = pd_data.loc[:, 'charges']
        self.x_train, self.x_test, self.y_train, self.y_test = \
            train_test_split(self.pd_x, self.pd_y, test_size=0.2, random_state=532)
        self.score = 0

    def build_model(self):
        '''
        build the model
        '''
        l_regression = LinearRegression()
        model = l_regression.fit(self.x_train, self.y_train)
        self.score = l_regression.score(self.x_test, self.y_test)
        print('model para:')
        print(model)
        print('intercept:')
        print(l_regression.intercept_)
        print('coef:')
        print(l_regression.coef_)
        print('score', self.score)

        y_pred = l_regression.predict(self.x_test)
        plt.figure()
        plt.show()
        plt.plot(range(len(y_pred)), y_pred, 'b', label="predict")
        plt.plot(range(len(y_pred)), self.y_test, 'r', label="test")
        plt.legend(loc="upper right")
        plt.xlabel("number ")
        plt.ylabel('charges')
        plt.show()

    def compared(self):
        '''
        compared result
        '''
        pd_data = pd.read_csv('insurance2.csv')
        sam = []
        array = ['age', 'bmi', 'children', 'charges', 'region', 'sex', 'smoker']
        dic = {}
        for i in array:
            elm = pd_data.loc[:, i]
            dic[i] = list(elm)
        print(dic)
        for i in range(len(dic['charges'])):
            result = 933217481 + float(dic['age'][i]) * 0.30861709 + float(
                dic['sex'][i]) * -0.00158138 + float(
                dic['bmi'][i]) * 0.170 - 0.00514220902337 + float(dic['children'][i]) * (
                    0.04348952) + float(dic['region'][i]) * -0.03114971 + float(dic['smoker'][i] * 0.78704948)
            sam.append(result)

        with open('insurance3.csv', 'w') as file:
            writer = csv.writer(file)
            writer.writerow(['charges', 'Predictive value'])
            for i in enumerate(sam):
                writer.writerow([dic['charges'][i], sam[i]])
        print('done')
        pd_data = pd.read_csv('insurance3.csv')
        pd_data.plot()
        plt.show()
