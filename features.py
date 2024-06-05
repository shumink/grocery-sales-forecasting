import pandas as pd
import numpy as np
import os
import io
import py7zr
import pickle
import math
import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from joblib import Parallel, delayed
from sklearn.preprocessing import OrdinalEncoder
from datetime import datetime, timedelta

DATASET_PATH = 'dataset'

def read_csv_from_7z(path):
    with py7zr.SevenZipFile(path, mode='r') as f:
        extracted_file = f.readall()
    file_name = list(extracted_file.keys())[0]
    data = extracted_file[file_name].read()
    return pd.read_csv(io.BytesIO(data))

def _calculate_moving_average(df, group_cols, window, col_name):
    df = df.sort_values(group_cols)
    id_cols = [e for e in group_cols if e != 'date']
    rolling_mean = df.groupby(group_cols).sum()['unit_sales'].rolling(window, min_periods=1).mean().unstack(id_cols).shift(1,freq="D").stack(id_cols)
    
    rolling_mean = rolling_mean.reset_index()
    return rolling_mean.rename(columns={0: col_name}).fillna(0).set_index(group_cols)


class FeatureService:
    def __init__(self, join_key, data_path=None) -> None:
        if data_path:
            self.df = read_csv_from_7z(data_path)
        self.join_key = join_key
        self.categorical = []
        self.continuous = []
    
    def join(self, spine) -> pd.DataFrame:
        res = spine.merge(self.df, how='left', on=self.join_key)
        # for feat in self.categorical:
        #     res[feat] = res[feat].fillna('null')
        return res

class StoreFeatureService(FeatureService):
    def __init__(self,) -> None:
        super().__init__(["store_nbr"], os.path.join(DATASET_PATH, f"stores.csv.7z"))
        self.df['store_type'] = self.df['type']
        self.df = self.df.drop(['type'], axis=1)
        self.categorical = ['store_nbr', 'city', 'state', 'cluster', 'store_type']

class ItemFeatureService(FeatureService):
    def __init__(self) -> None:
        super().__init__(["item_nbr"], os.path.join(DATASET_PATH, f"items.csv.7z"))
        self.categorical = [
            'item_nbr', 'family', 'class', 'perishable'
        ]

class HolidayFeatureService(FeatureService):
    def __init__(self) -> None:
        super().__init__(["date"], os.path.join(DATASET_PATH, f"holidays_events.csv.7z"))
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df['holiday_type'] = self.df['type']
        self.df = self.df.drop(['type'], axis=1)
        self.categorical = ['holiday_type', 'locale', 'locale_name', 'description', 'transferred']


class OilFeatureService(FeatureService):
    def __init__(self) -> None:
        super().__init__(["date"], os.path.join(DATASET_PATH, f"oil.csv.7z"))
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.continuous = ['dcoilwtico']

class MovingAverageFeatureService(FeatureService):
    def __init__(self, join_key, window) -> None:
        super().__init__(join_key)
        self.window = window
        self.continuous = [f"ma_{window}d_({'_'.join(join_key)})"]

    def join(self, spine) -> pd.DataFrame:
        self.df = _calculate_moving_average(spine, self.join_key, self.window, self.continuous[0])
        return spine.merge(self.df, how='left', on=self.join_key)

class DateFeatureService(FeatureService):
    def __init__(self) -> None:
        super().__init__("date")
        self.categorical = [    
            'month', 
            'day_of_month', 
            'day_of_week', 
        ]
        self.continuous = ['year']

    def join(self, spine) -> pd.DataFrame:
        spine['year'] = spine[self.join_key].dt.year
        spine['month'] = spine[self.join_key].dt.month
        spine['day_of_month'] = spine[self.join_key].dt.day
        spine['day_of_week'] = spine[self.join_key].dt.dayofweek

        return spine
