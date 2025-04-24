#%%
# import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import datetime

directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '../gemini_crypto_usd_csv/'))
data_files = os.listdir(directory)
column_names = ['unix', 'date', 'symbol', 'open', 'high', 'low', 'close', 'Volume BTC', 'Volume USD']


#%%
for file in data_files:
    data = pd.read_csv(os.path.join(directory, file), names=column_names, header=0, low_memory=False)
    for col in data.columns:
        print(data[col])
    #print(data.columns)
    plt.plot(data['unix'][0, 100], data['close'][0, 100])
    plt.show()

#%%
for file in data_files:
    data = pd.read_csv(os.path.join(directory, file), names=column_names, header=0, low_memory=False)
    for col in data.columns:
        print(data[col])
    print(data.columns)
    plt.figure(figsize=(12, 6))
    plt.plot(data['unix'][:1000], data['high'][:1000])
    plt.xlabel('Unix Timestamp')
    plt.ylabel('Close Price')
    plt.title('Close Price Over Time (First 100 rows)')
    plt.show()

# %%
for file in data_files:
    data = pd.read_csv(os.path.join(directory, file), names=column_names, header=0, low_memory=False)
    data['unix'] = pd.to_datetime(data['unix'], unit='ms')
    data['date'] = data['unix'].dt.strftime('%b-%y')
    
    plt.figure(figsize=(12, 6))
    plt.plot(data['unix'][:100], data['close'][:100])
    
    # Set the x-axis tick labels to the 'date' column
    plt.xticks(data['unix'][:100:10], data['date'][:100:10], rotation=45)
    
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.title('Close Price Over Time (First 100 rows)')
    plt.show()
# %%
