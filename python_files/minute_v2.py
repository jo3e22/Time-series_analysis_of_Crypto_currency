import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.signal import correlate
from statsmodels.tsa.stattools import grangercausalitytests
import smoothing as mySm # Custom smoothing module

directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '../ftx_minute_csv/'))
data_files = os.listdir(directory)
column_names = ['unix', 'date', 'symbol', 'open', 'high', 'low', 'close', 'Volume BTC', 'Volume USD']

arr_o_data = []
for file in data_files:
    data = pd.read_csv(os.path.join(directory, file), names=column_names, header=0, low_memory=False)
    data['unix'] = pd.to_datetime(data['unix'], unit='ms')
    arr_o_data.append(data)

# Resampling and Visualization
def resample_and_plot(dfs, resample_freq):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    for df in dfs:
        symbol = df['symbol'].iloc[0]
        resampled = df.resample(resample_freq, on='unix').last()
        ax1.plot(resampled.index, resampled['close'], label=symbol)
        ax2.plot(resampled.index, resampled['Volume USD'], label=symbol, alpha=0.5)
    ax1.set_title(f"{resample_freq} Resampled Crypto Prices")
    ax2.set_title(f"{resample_freq} Resampled Crypto Volumes")
    ax1.legend()
    ax2.legend()
    plt.show()

resample_and_plot(arr_o_data, '5T')
resample_and_plot(arr_o_data, '15T')
resample_and_plot(arr_o_data, '30T')

# Correlation Analysis
def calculate_correlation(dfs, resample_freq):
    resampled_dfs = [df.resample(resample_freq, on='unix').last() for df in dfs]
    corr_matrix = pd.DataFrame(index=[df['symbol'].iloc[0] for df in resampled_dfs], columns=[df['symbol'].iloc[0] for df in resampled_dfs])
    for i, df1 in enumerate(resampled_dfs):
        for j, df2 in enumerate(resampled_dfs):
            corr_matrix.iloc[i, j] = df1['close'].corr(df2['close'])
    return corr_matrix

print("Correlation Matrix (5-minute):")
print(calculate_correlation(arr_o_data, '5T'))
print("Correlation Matrix (15-minute):")
print(calculate_correlation(arr_o_data, '15T'))
print("Correlation Matrix (30-minute):")
print(calculate_correlation(arr_o_data, '30T'))

# Time Lag Analysis
def calculate_time_lag(df1, df2):
    corr = correlate(df1['close'], df2['close'], mode='full')
    lags = np.arange(-len(df1) + 1, len(df2))
    max_corr_idx = np.argmax(corr)
    max_corr = corr[max_corr_idx]
    time_lag = lags[max_corr_idx]
    return time_lag, max_corr

time_lags = {}
for i in range(len(arr_o_data)):
    for j in range(i + 1, len(arr_o_data)):
        lag, corr = calculate_time_lag(arr_o_data[i].resample('5T', on='unix').last(), arr_o_data[j].resample('5T', on='unix').last())
        time_lags[(arr_o_data[i]['symbol'].iloc[0], arr_o_data[j]['symbol'].iloc[0])] = (lag, corr)

print("Time Lags:")
for key, value in time_lags.items():
    print(f"{key[0]} vs {key[1]}: Lag = {value[0]} hours, Correlation = {value[1]}")

# Robustness to Data Gaps
def introduce_periodic_gaps(df, gap_length=24, gap_frequency=7):
    new_df = df.copy()
    for i in range(0, len(new_df), gap_frequency * 24):
        new_df = new_df.drop(new_df.index[i:i+gap_length])
    return new_df

def introduce_non_periodic_gaps(df, gap_probability=0.1):
    new_df = df.copy()
    mask = np.random.rand(len(new_df)) < gap_probability
    new_df = new_df.loc[~mask]
    return new_df
def introduce_non_periodic_gaps(df, gap_probability=0.1):
    new_df = df.copy()
    mask = np.random.rand(len(new_df)) < gap_probability
    new_df = new_df.loc[~mask]
    return new_df

dfs_with_gaps = []
for df in arr_o_data:
    dfs_with_gaps.append(introduce_periodic_gaps(df))
    dfs_with_gaps.append(introduce_non_periodic_gaps(df))

print("Correlation Matrix with Gaps:")
corr_matrix_gaps = calculate_correlation(dfs_with_gaps)
print(corr_matrix_gaps)

print("Time Lags with Gaps:")
time_lags_gaps = {}
for i in range(len(dfs_with_gaps)):
    for j in range(i + 1, len(dfs_with_gaps)):
        lag, corr = calculate_time_lag(dfs_with_gaps[i], dfs_with_gaps[j])
        time_lags