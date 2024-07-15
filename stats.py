import os
import functools
import pandas as pd
import numpy as np

class StatsCalculator:
    def __init__(self, stats_dir):
        self.stats_dir = stats_dir
    
    @functools.cache
    def _get_csv(self):
        files = os.listdir(self.stats_dir)
        return [file for file in files if file.endswith(".csv")]

    def _get_num_tables(self):
        return len(self._get_csv())

    def _get_sparsity(self):
        files = self._get_csv()
        sparsity = []
        for file in files:
            try:
                df = pd.read_csv(os.path.join(self.stats_dir, file))
                df = df.replace('', np.nan)
                if df.isna().all().all():
                    continue
                num_nans = df.isna().sum().sum()
                sparsity_score = num_nans / (df.shape[0] * df.shape[1])
                sparsity.append(sparsity_score)
            except:
                print(f"error with file {file}")
        
        return round(float(sum(sparsity) / len(sparsity)), 4)

    def run(self):
        print(f"The number of tables extracted is {self._get_num_tables()}")
        print(f"The mean sparsity of the tables is {self._get_sparsity()}")