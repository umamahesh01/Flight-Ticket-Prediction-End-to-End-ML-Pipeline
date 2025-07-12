import pandas as pd
from sklearn.preprocessing import StandardScaler   

class Scaling:

    def __init__(self):
        self.scaled = StandardScaler()

    def scale(self, df: pd.DataFrame, cols : list) -> pd.DataFrame:
        df = df.copy()

        df[cols] = self.scaled.fit_transform(df[cols])
        return df
    
    def transform(self, df: pd.DataFrame, cols: list) -> pd.DataFrame:      # for unseen test data
        df = df.copy()

        df[cols] = self.scaled.transform(df[cols])
        return df