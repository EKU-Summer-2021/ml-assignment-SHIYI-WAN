import unittest
import pandas as pd
from src.linear_regression import load_data, normalization, encode, LinearRegressor


class LinearRegressionTest(unittest.TestCase):
  '''
  class to test linearRegressor
  '''
    def test_encode(self):
        '''
        unittest for function encode
        '''
        data = encode()
        d = data.iloc[0, 5]
        self.assertEqual(d, 0.0)

    def test_normaliztion(self):
        '''
        unittest for function normalization
        '''
        normalization()
        d = pd.read_csv('insurance2.csv')
        self.assertIsInstance(pd.DataFrame, d)

    def set_up(self):
        self.lr = LinearRegressor()

    def test_build_model(self):
        '''
        unittest for build_model fuc
        '''
        self.lr.build_model()
        self.assertNotEqual(0, self.lr.score)

    def test_compare(self):
        '''
        unittest for compare fuc
        '''
        data = pd.read_csv('insurance3.csv')
        self.lr.compared()
        self.assertIsNotNone(data)
