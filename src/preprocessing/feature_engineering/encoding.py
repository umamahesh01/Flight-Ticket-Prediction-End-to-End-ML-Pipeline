import pandas as pd
import pickle
import os
from sklearn.preprocessing import LabelEncoder


class LabelEncoding:
    def __init__(self):
        self.encoders = {}  # store encoders for each column

    def encode(self, df: pd.DataFrame, cols: list) -> pd.DataFrame:
        df = df.copy()
        
        for col in cols:
            le = LabelEncoder()
            if df[col].dtype == 'object':   # ensuring to do encoding only on categorical columns
                df[col] = le.fit_transform(df[col].astype(str))
                self.encoders[col] = le  # store this column's encoder
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:   # for unseen test data
        df = df.copy()

        for col, le in self.encoders.items():
            df[col] = le.transform(df[col])

        return df
    
    # saving encoder
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)  
        with open(path, 'wb') as f:
            pickle.dump(self.encoders, f)
    