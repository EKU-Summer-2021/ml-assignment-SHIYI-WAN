'''
module for linear regression
'''
import urllib.request
import pandas as pd


def load_data(path, filename):
    '''
    function load data from url
    '''
    urllib.request.urlretrieve(path, filename)
    data = pd.read_csv(filename)
    return data

def normalization():
    '''
    normalization data
    '''
    pd_data = load_data("https://github.com/stedy/Machine-Learning-with-R-datasets"
                        "/blob/master/insurance.csv", "insurance.csv")
    sam = []
    a = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
    for i in a:
        y = pd_data.loc[:, i]
        ys = list(preprocessing.scale(y))
        sam.append(ys)

    print(len(sam))
    with open('insurance2.csv', 'w') as file:
        writer = csv.writer(file)
        for i in range(len(sam[0])):
            writer.writerow([sam[0][i], sam[1][i], sam[2][i], sam[3][i], sam[4][i], sam[5][i]])
    print("done")

class LinearRegression:
    '''
    create a class to initialize Linear Regression models
    '''
    def __init__(self):
        '''
        set the configuration of the Linear Regression model with a constructor.
        '''
        pd_data = pd.read_csv('insurance2.csv')
        self.X = pd_data.loc[:, ('age', 'sex', 'bmi', 'children', 'region')]
        self.y = pd_data.loc[:, 'charges']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=532)
