#%%
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
def resample_and_plot(dfs, lambda_=10):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 16))
    for df in dfs:
        df = df.copy()
        df = df[0:5000]

        symbol = df['symbol'].iloc[0]
        
        df.loc[:, 'pc close'] = df['close'].pct_change()
        df.loc[0, 'pc close'] = 0
        df.loc[:, 'pc open'] = df['open'].pct_change()
        df.loc[:, 'pc Volume USD'] = df['Volume USD'].pct_change()
        df.loc[0, 'pc Volume USD'] = 0

        df['z close'] = (  ( df['close']/np.max(df['close']) )  -  ( df.loc[0, 'close']/np.max(df['close']) )  )

        df['smoothed z close'] = mySm.penalized_least_squares(df['z close'], lambda_)
        #df['smoothed pc close'] = mySm.penalized_least_squares(df['pc close'], lambda_)
        df['smoothed volume'] = mySm.penalized_least_squares(df['Volume USD'], lambda_)

        fft_signal = np.fft.fft(df['smoothed volume'])
        fft_freq = np.fft.fftfreq(len(df), d=1)
        fft_signal = np.fft.fftshift(fft_signal)
        fft_freq = np.fft.fftshift(fft_freq)

        fft_signal_close = np.fft.fft(df['smoothed z close'])
        fft_freq_close = np.fft.fftfreq(len(df), d=1)
        fft_signal_close = np.fft.fftshift(fft_signal_close)
        fft_freq_close = np.fft.fftshift(fft_freq_close)

        ax1.plot(df.index, df['smoothed z close'], label=f'smth z cls: {symbol}')
        ax2.plot(df.index, df['smoothed volume'], label=f'smth vol: {symbol}', alpha=0.5)
        ax3.plot(fft_freq, np.abs(fft_signal), label=f'fft: {symbol}', alpha=0.5)
        ax4.plot(fft_freq_close, np.abs(fft_signal_close), label=f'fft: {symbol}', alpha=0.5)

    ax1.set_title(f"{lambda_} Smoothed Crypto Prices")
    ax2.set_title(f"{lambda_} Smoothed Crypto Volumes")
    ax3.set_title(f"FFT of Crypto Volumes")
    ax4.set_title(f"FFT of Crypto Prices")
    ax1.set_xlabel('Time')
    ax2.set_xlabel('Time')
    ax3.set_xlabel('Frequency')
    ax4.set_xlabel('Frequency')
    ax1.set_ylabel('Price')
    ax2.set_ylabel('Volume')
    ax3.set_ylabel('Magnitude')
    ax4.set_ylabel('Magnitude')

    ax3.set_xscale('log')
    ax4.set_xscale('log')

    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()
    plt.show()



#resample_and_plot(arr_o_data, 5)
#resample_and_plot(arr_o_data, 10)
#resample_and_plot(arr_o_data, 15)
resample_and_plot(arr_o_data, 20)

# %%
