import pandas as pd
from sklearn.preprocessing import LabelEncoder


class LabelEncoding:
    def __init__(self):
        self.encoders = {}  # store encoders for each column

    def encode(self, df: pd.DataFrame, cols: list) -> pd.DataFrame:
        df = df.copy()
        
        for col in cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.encoders[col] = le  # store this column's encoder
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:   # for unseen test data
        df = df.copy()

        for col, le in self.encoders.items():
            df[col] = le.transform(df[col])

        return df