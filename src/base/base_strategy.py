from abc import ABC, abstractmethod
import pandas as pd

class MissingValueStrategy(ABC):
    @abstractmethod
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        pass
