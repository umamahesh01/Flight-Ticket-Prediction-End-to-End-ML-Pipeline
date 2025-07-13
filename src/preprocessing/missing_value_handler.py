import pandas as pd
from src.base.base_strategy import MissingValueStrategy

class FillMissingValuesStrategy(MissingValueStrategy):

    def __init__(self, num_method='mean', cat_method='mode', fill_value=None):
        self.num_method = num_method
        self.cat_method = cat_method
        self.fill_value = fill_value

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()  # Works on a copy() of the DataFrame to avoid modifying original

        # Handling Categorical cols
        for i in df.select_dtypes(include= 'object').columns:
            if self.cat_method == 'mode':
                df[i].fillna(df[i].mode()[0])
            elif self.cat_method == 'constant':
                df[i] = df[i].fillna(self.fill_value)
            else:
                raise ValueError(f"{self.method} not valid for categorical columns")

        # Handling numerical cols   
        for i in df.select_dtypes(exclude='object').columns:

            if self.num_method == 'mean':
                df[i] = df[i].fillna(df[i].mean())
            elif self.num_method == 'mode':
                df[i].fillna(df[i].mode()[0])
            elif self.num_method == 'median':
                df[i].fillna(df[i].median())
            elif self.num_method == "constant":
                df.fillna(self.fill_value)
            else:
                raise ValueError("Invalid method for numerical columns")
            
        return df

