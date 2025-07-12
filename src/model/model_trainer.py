from sklearn.linear_model import LinearRegression
import pandas as pd
import os
import pickle

class LinearModel :

    def __init__(self):
        self.model = LinearRegression()

    def train(self,x,y):

        self.model.fit(x,y)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)  
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)


