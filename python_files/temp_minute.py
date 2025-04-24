#%%
# import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import datetime
import numpy as np

directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '../ftx_minute_csv/'))
data_files = os.listdir(directory)
column_names = ['unix', 'date', 'symbol', 'open', 'high', 'low', 'close', 'Volume BTC', 'Volume USD']

# %%
arr_o_data = []
for file in data_files:
    data = pd.read_csv(os.path.join(directory, file), names=column_names, header=0, low_memory=False)
    arr_o_data.append(data)
# %%
fig = plt.figure(figsize=(8, 4))
axClose = fig.add_subplot(211)
axVol = fig.add_subplot(212)
for i, data in enumerate(arr_o_data):
    plot_fft(data, f'FFT of {data["symbol"].iloc[0]}')
    data['unix'] = pd.to_datetime(data['unix'], unit='ms')
    data['date'] = data['unix'].dt.strftime('%b-%y')

    data['close_zeroed'] = data['close'] - data['close'].iloc[1]
    data['close_zeroed'] = data['close_zeroed'] / data['close_zeroed'].max()
    data['close_zeroed'] = data['close_zeroed'] * 100
    axClose.plot(data['unix'], data['close_zeroed'], label=f'Close price of data {data['symbol'].iloc[0]} (relative)')
    axVol.plot(data['unix'], data['Volume USD'], label=f'Volume of {data["symbol"].iloc[0]} (USD)', alpha=0.5)

axVol.legend()
axClose.legend()
axClose.xaxis.set_visible(False)
axVol.tick_params(axis='x', rotation=45)
# %%

#fast fourier transform and plotting
def plot_fft(data, title):
    data['fft'] = np.fft.fft(data['close'])
    data['freq'] = np.fft.fftfreq(len(data), d=1/60)  # Assuming data is sampled every minute
    plt.figure(figsize=(12, 6))
    plt.plot(data['freq'], np.abs(data['fft']))
    plt.title(title)
    plt.xlabel('Frequency')
    plt.ylabel('Magnitude')
    plt.xlim(0, 0.02)
    plt.grid()
    plt.show()
# %%
