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
