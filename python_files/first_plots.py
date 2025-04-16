#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data_folder = '../gemini_crypto_usd_csv/'
# format: unix,date,symbol,open,high,low,close,Volume BTC,Volume USD

#%%
for file in data_folder:
    data = pd.read_csv(file)
    print(data.head())
    print(data.info())
    print(data.describe())
    print(data.columns)
    print(data.dtypes)
    print(data.isnull().sum())
    print(data.isna().sum())


