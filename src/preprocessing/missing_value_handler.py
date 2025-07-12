import pandas as pd
from src.base.base_strategy import MissingValueStrategy

class FillMissingValuesStrategy(MissingValueStrategy):

    def __init__(self, method = 'mean', fill_value = None) :
        self.method = method
        self.fill_value = fill_value

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()  # Works on a copy() of the DataFrame to avoid modifying original

        for i in df.select_dtypes(include= 'object').columns:
            if self.method == 'mode':
                df[i].fillna(df[i].mode()[0])
            elif self.method == 'constant':
                df[i] = df[i].fillna(self.fill_value)
            else:
                raise ValueError(f"{self.method} not valid for categorical columns")
            
        for i in df.select_dtypes(exclude='object').columns:

            if self.method == 'mean':
                df[i] = df[i].fillna(df[i].mean())

            elif self.method == 'mode':
                df[i].fillna(df[i].mode()[0])
            elif self.method == 'median':
                df[i].fillna(df[i].median())
            elif self.method == "constant":
                df.fillna(self.fill_value)
            else:
                raise ValueError("Invalid method for numerical columns")
            
        return df

