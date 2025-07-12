import pandas as pd

class DataLoader:

    def __init__(self, path : str) :
        self.path = path

    def load(self) -> pd.DataFrame:
        return pd.read_csv(self.path)
