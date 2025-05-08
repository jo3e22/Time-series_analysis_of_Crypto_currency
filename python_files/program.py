'''
program.py
Author:     James O'Donohoe
Date:       06/05/2025
Contact:    james@odonohoe.ie

Comments: CT assignment 2, This is my own work though genAI has been used to assist with formating, commenting, and troubleshooting.
'''


#%%  Import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.signal import correlate
from scipy import stats
import smoothing as mySm  # Custom smoothing module


#%%  Set up data directory
saveDir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../savedFigs/'))  # for saved outputs
directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '../ftx_minute_csv/'))  # for data files
data_files = os.listdir(directory)
column_names = ['unix', 'date', 'symbol', 'open', 'high', 'low', 'close', 'Volume BTC', 'Volume USD']


#%% Load data
arr_o_data = []  # initialise data array
for file in data_files:
    data = pd.read_csv(os.path.join(directory, file), names=column_names, header=0, low_memory=False)
    data['unix'] = pd.to_datetime(data['unix'], unit='ms')
    arr_o_data.append(data)


#%% Process data and create a merged dataframe for analysis
def process_and_merge_data(data_array, max_rows=None):
    """
    Function: Process and merge cryptocurrency data for analysis
    Inputs: 
        data_array: List of dataframes containing cryptocurrency data
        max_rows: Maximum number of rows to process from each dataframe (optional)
    Outputs:
        merged_df: Merged dataframe with datetime as index and closing prices for each cryptocurrency
    """
    processed_dfs = []
    
    # Process each dataframe
    for df in data_array:
        df = df.copy()
        if max_rows:  # Limit the number of rows if specified
            df = df.head(max_rows)
        
        # Get symbol and prepare dataframe
        symbol = df['symbol'].iloc[0]
        
        # Set datetime as index
        df.set_index('unix', inplace=True)
        
        # Select only the closing price and rename it to the symbol
        price_df = df[['close']].rename(columns={'close': symbol})
        processed_dfs.append(price_df)
    
    # Merge all dataframes on datetime index
    merged_df = pd.concat(processed_dfs, axis=1)
    
    # Remove any rows with missing values
    merged_df = merged_df.dropna()
    
    return merged_df


#%% Function to normalize and smooth data
def normalize_and_smooth(df, lambda_=10):
    """
    Function: Normalize and smooth the data
    Inputs:
        df: DataFrame containing the cryptocurrency data
        lambda_: Smoothing parameter for penalized least squares
    Outputs:
        normalized_df: DataFrame with normalized and smoothed data
        norm_columns: List of normalized column names
        smooth_columns: List of smoothed column names
    """
    normalized_df = df.copy()
    
    # Normalize each column (each cryptocurrency)
    for column in normalized_df.columns:
        # Z-score normalization, transforms data to have mean=0 and std=1
        normalized_df[f"{column}_norm"] = (normalized_df[column] - normalized_df[column].mean()) / normalized_df[column].std()
        
        # Apply smoothing using custom smoothing function (seperate module for modularity)
        normalized_df[f"{column}_smooth"] = mySm.penalized_least_squares(normalized_df[f"{column}_norm"], lambda_)
    
    # Keep only normalized and smoothed columns
    norm_columns = [col for col in normalized_df.columns if '_norm' in col]
    smooth_columns = [col for col in normalized_df.columns if '_smooth' in col]
    
    return normalized_df, norm_columns, smooth_columns


#%% Plot time series
def plot_time_series(df, columns, title="Cryptocurrency Price Movement"):
    """
    Function: Plot multiple time series
    Inputs:
        df: DataFrame containing the data to plot
        columns: List of column names to plot
        title: Title of the plot
    Outputs:
        None (saves the plot to the specified directory)
    """
    plt.figure(figsize=(12, 6))
    
    for column in columns:
        plt.plot(df.index, df[column], label=column)
    
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Normalized Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(saveDir, 'crypto_time_series.png'), dpi=300)
    plt.show()


#%% Calculate correlations
def calculate_correlations(df, columns):
    """
    Function: Calculate correlation matrix for selected columns
    Inputs:
        df: DataFrame containing the data
        columns: List of column names to calculate correlations for
    Outputs:
        corr_matrix: DataFrame containing the Pearson correlation matrix
        spearman_corr: DataFrame containing the Spearman correlation matrix
    """
    corr_matrix = df[columns].corr(method='pearson')
    spearman_corr = df[columns].corr(method='spearman')
    
    return corr_matrix, spearman_corr


#%% Plot correlation heatmap
def plot_correlation_heatmap(corr_matrix, title="Correlation Matrix"):
    """
    Function: Plot correlation heatmap
    Inputs:
        corr_matrix: DataFrame containing the correlation matrix
        title: Title of the plot
    Outputs:
        None (saves the plot to the specified directory)
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(saveDir, f"{title.lower().replace(' ', '_')}.png"), dpi=300)
    plt.show()


#%% Calculate cross-correlation for time lag analysis
def calculate_time_lag(series1, series2, max_lag=100):
    """
    Function: Calculate time lag between two time series using cross-correlation
    Inputs:
        series1: First time series (numpy array)
        series2: Second time series (numpy array)
        max_lag: Maximum lag to consider for cross-correlation
    Outputs:
        optimal_lag: Optimal lag value
        max_corr_value: Maximum correlation value
        cross_corr: Cross-correlation values
        lag_array: Array of lag values
    """
    # Compute cross-correlation
    cross_corr = correlate(series1, series2, mode='full') / len(series1)
    
    # Get lag array
    lag_array = np.arange(-len(series1) + 1, len(series2))
    
    # Find lag with maximum correlation
    max_corr_idx = np.argmax(np.abs(cross_corr))
    optimal_lag = lag_array[max_corr_idx]
    max_corr_value = cross_corr[max_corr_idx]
    
    return optimal_lag, max_corr_value, cross_corr, lag_array


#%% Bootstrap analysis for confidence intervals
def bootstrap_lag_confidence(series1, series2, n_bootstrap=1000, confidence=0.95):
    """
    Function: Use bootstrap to calculate confidence intervals for time lag estimates
    Inputs:
        series1: First time series (numpy array)
        series2: Second time series (numpy array)
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level for the intervals
    Outputs:
        mean_lag: Mean lag from bootstrap samples
        lower_bound: Lower bound of the confidence interval
        upper_bound: Upper bound of the confidence interval
    """
    n = len(series1)
    lag_results = []
    
    np.random.seed(42)  # For reproducibility
    
    for _ in range(n_bootstrap):
        # Generate bootstrap sample indices
        indices = np.random.choice(range(n), size=n, replace=True)
        
        # Get bootstrap samples
        boot_series1 = series1[indices]
        boot_series2 = series2[indices]
        
        # Calculate lag for this bootstrap sample
        lag, _, _, _ = calculate_time_lag(boot_series1, boot_series2)
        lag_results.append(lag)
    
    # Calculate confidence interval
    lower_bound = np.percentile(lag_results, (1 - confidence) * 100 / 2)
    upper_bound = np.percentile(lag_results, 100 - (1 - confidence) * 100 / 2)
    
    return np.mean(lag_results), lower_bound, upper_bound


#%% Plot cross-correlation
def plot_cross_correlation(cross_corr, lag_array, series1_name, series2_name, optimal_lag):
    global saveDir
    """
    Function: Plot cross-correlation function
    Inputs:
        cross_corr: Cross-correlation values
        lag_array: Array of lag values
        series1_name: Name of the first time series
        series2_name: Name of the second time series
        optimal_lag: Optimal lag value
    Outputs:
        None (saves the plot to the specified directory)
    """
    plt.figure(figsize=(12, 6))
    plt.plot(lag_array, cross_corr)
    plt.axvline(x=optimal_lag, color='red', linestyle='--', 
                label=f'Optimal lag: {optimal_lag}')
    plt.axvline(x=0, color='green', linestyle=':', label='Zero lag')
    
    plt.title(f'Cross-correlation between {series1_name} and {series2_name}')
    plt.xlabel('Lag')
    plt.ylabel('Cross-correlation')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    filename = f'cross_corr_{series1_name}_{series2_name}.png'
    filepath = os.path.join(saveDir, filename)
    plt.savefig(filepath, dpi=300)
    plt.show()


#%% Introduce data gaps
def introduce_gaps(series, gap_percentage, periodic=True):
    """
    Function: Introduce gaps in time series data
    Inputs:
        series: Time series data (numpy array)
        gap_percentage: Percentage of data to be replaced with NaN
        periodic: If True, gaps are periodic; if False, gaps are random
    Outputs:
        gapped_series: Time series with gaps
        gap_indices: Indices of the introduced gaps
    """
    series_copy = series.copy()
    n = len(series_copy)
    gap_count = int(n * gap_percentage)
    
    if periodic:
        # Create periodic gaps
        gap_period = n // gap_count
        gap_indices = [i * gap_period for i in range(gap_count) if i * gap_period < n]
    else:
        # Create random gaps
        np.random.seed(42)  # For reproducibility
        gap_indices = np.random.choice(range(n), gap_count, replace=False)
    
    # Create a new series with gaps
    gapped_series = series_copy.copy()
    gapped_series[gap_indices] = np.nan
    
    return gapped_series, gap_indices


#%% Analyze impact of gaps on correlation
def analyse_gap_impact(df, columns, gap_percentages=[0.05, 0.1, 0.2, 0.3]):
    """
    Function: Analyze how data gaps affect correlation and lag detection
    Inputs:
        df: DataFrame containing the data
        columns: List of column names to analyze
        gap_percentages: List of gap percentages to test
    Outputs:
        results: Dictionary containing correlation and lag results for each gap percentage
    """
    results = {}
    base_corr = df[columns].corr().iloc[0, 1]  # Correlation between first two columns
    
    series1 = df[columns[0]].values
    series2 = df[columns[1]].values
    
    # Calculate original time lag
    original_lag, _, _, _ = calculate_time_lag(series1, series2)
    
    for gap_pct in gap_percentages:
        # Results for this gap percentage
        results[gap_pct] = {}
        
        # Periodic gaps
        series1_periodic, _ = introduce_gaps(series1, gap_pct, periodic=True)
        # Filter out NaN values for correlation calculation
        mask_periodic = ~np.isnan(series1_periodic)
        if sum(mask_periodic) > 1:  # Need at least 2 points for correlation
            corr_periodic = np.corrcoef(series1_periodic[mask_periodic], 
                                        series2[mask_periodic])[0, 1]
            # Fill NaNs with interpolation for lag calculation
            series1_periodic_filled = pd.Series(series1_periodic).interpolate().values
            lag_periodic, _, _, _ = calculate_time_lag(series1_periodic_filled, series2)
        else:
            corr_periodic = np.nan
            lag_periodic = np.nan
            
        # Random gaps
        series1_random, _ = introduce_gaps(series1, gap_pct, periodic=False)
        # Filter out NaN values for correlation calculation
        mask_random = ~np.isnan(series1_random)
        if sum(mask_random) > 1:  # Need at least 2 points for correlation
            corr_random = np.corrcoef(series1_random[mask_random], 
                                      series2[mask_random])[0, 1]
            # Fill NaNs with interpolation for lag calculation
            series1_random_filled = pd.Series(series1_random).interpolate().values
            lag_random, _, _, _ = calculate_time_lag(series1_random_filled, series2)
        else:
            corr_random = np.nan
            lag_random = np.nan
        
        results[gap_pct] = {
            'periodic_gaps_corr': corr_periodic,
            'random_gaps_corr': corr_random,
            'periodic_gaps_lag': lag_periodic,
            'random_gaps_lag': lag_random,
            'original_corr': base_corr,
            'original_lag': original_lag
        }
    
    return results


#%% Plot gap impact results
def plot_gap_impact(gap_results, metric='corr'):
    """
    Function: Plot impact of gaps on correlation or lag detection
    Inputs:
        gap_results: Dictionary containing results from gap analysis
        metric: 'corr' for correlation or 'lag' for time lag
    Outputs:
        None (saves the plot to the specified directory)
    """
    global saveDir

    plt.figure(figsize=(10, 6))
    
    gap_percentages = list(gap_results.keys())
    
    if metric == 'corr':
        periodic_values = [gap_results[g]['periodic_gaps_corr'] for g in gap_percentages]
        random_values = [gap_results[g]['random_gaps_corr'] for g in gap_percentages]
        original_value = gap_results[gap_percentages[0]]['original_corr']
        y_label = 'Correlation Coefficient'
        title = 'Impact of Data Gaps on Correlation Detection'
        filename = 'gap_impact_correlation.png'
    else:  # lag
        periodic_values = [gap_results[g]['periodic_gaps_lag'] for g in gap_percentages]
        random_values = [gap_results[g]['random_gaps_lag'] for g in gap_percentages]
        original_value = gap_results[gap_percentages[0]]['original_lag']
        y_label = 'Detected Time Lag'
        title = 'Impact of Data Gaps on Time Lag Detection'
        filename = 'gap_impact_lag.png'
    
    plt.plot(gap_percentages, periodic_values, 'o-', label='Periodic Gaps')
    plt.plot(gap_percentages, random_values, 's-', label='Random Gaps')
    plt.axhline(y=original_value, color='red', linestyle='--', label='Original Value')
    
    plt.legend()
    plt.title(title)
    plt.xlabel('Gap Percentage')
    plt.ylabel(y_label)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(saveDir, filename), dpi=300)
    plt.show()


#%% Main analysis function
def analyze_crypto_time_series(max_rows=5000, lambda_=10):
    """
    Function: Main function to run the complete analysis
    Inputs:
        max_rows: Maximum number of rows to process from each dataframe
        lambda_: Smoothing parameter for penalized least squares
    Outputs:
        results: Dictionary containing the results of the analysis
    """
    
    # Process and merge data
    print("Processing and merging data...")
    merged_df = process_and_merge_data(arr_o_data, max_rows)
    print(f"Merged dataframe shape: {merged_df.shape}")
    
    # Normalize and smooth data
    print("Normalizing and smoothing data...")
    normalized_df, norm_columns, smooth_columns = normalize_and_smooth(merged_df, lambda_)
    
    # Plot time series
    print("Plotting time series...")
    plot_time_series(normalized_df, norm_columns, "Normalized Cryptocurrency Prices")
    plot_time_series(normalized_df, smooth_columns, "Smoothed Cryptocurrency Prices")
    
    # Calculate correlations
    print("Calculating correlations...")
    #pearson_corr, spearman_corr = calculate_correlations(normalized_df, norm_columns)
    pearson_corr, spearman_corr = calculate_correlations(normalized_df, smooth_columns)
    
    # Plot correlation matrices
    print("Plotting correlation matrices...")
    plot_correlation_heatmap(pearson_corr, "Pearson Correlation Matrix")
    plot_correlation_heatmap(spearman_corr, "Spearman Correlation Matrix")
    
    # Time lag analysis
    print("Performing time lag analysis...")
    lag_results = {}
    crypto_names = [col.replace('_norm', '') for col in norm_columns]
    crypto_names = [name.replace('/', '.') for name in crypto_names]
    
    for i in range(len(norm_columns)):
        for j in range(i+1, len(norm_columns)):
            series1 = normalized_df[norm_columns[i]].values
            series2 = normalized_df[norm_columns[j]].values
            
            # Calculate time lag
            lag, corr_value, cross_corr, lag_array = calculate_time_lag(series1, series2)
            
            # Get bootstrap confidence intervals
            mean_lag, lower_bound, upper_bound = bootstrap_lag_confidence(series1, series2)
            
            # Store results
            pair_name = f"{crypto_names[i]}-{crypto_names[j]}"
            lag_results[pair_name] = {
                'lag': lag,
                'corr_value': corr_value,
                'mean_lag': mean_lag,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
            
            # Plot cross-correlation
            plot_cross_correlation(cross_corr, lag_array, crypto_names[i], 
                                  crypto_names[j], lag)
    
    # Print lag results
    print("\nTime Lag Results:")
    for pair, results in lag_results.items():
        print(f"{pair}: Optimal lag = {results['lag']}, 95% CI: [{results['lower_bound']:.2f}, {results['upper_bound']:.2f}]")
    
    # Analyze impact of gaps
    print("\nAnalyzing impact of data gaps...")
    # Use first two cryptocurrencies for gap analysis
    gap_results = analyse_gap_impact(normalized_df, norm_columns[:2])
    
    # Plot gap impact
    plot_gap_impact(gap_results, 'corr')
    plot_gap_impact(gap_results, 'lag')
    
    return {
        'merged_df': merged_df,
        'normalized_df': normalized_df,
        'pearson_corr': pearson_corr,
        'spearman_corr': spearman_corr,
        'lag_results': lag_results,
        'gap_results': gap_results
    }


#%% Run the analysis
if __name__ == "__main__":
    results = analyze_crypto_time_series(max_rows=15000, lambda_=50)
    
    # Display summary statistics for the report
    print("\nSummary Statistics for Report:")
    
    # Correlation summary
    print("\nCorrelation Summary:")
    print(results['pearson_corr'].round(3))
    
    # Time lag summary
    print("\nTime Lag Summary:")
    for pair, data in results['lag_results'].items():
        if data['lag'] > 0:
            leader, follower = pair.split('-')
            direction = f"{leader} leads {follower}"
        elif data['lag'] < 0:
            leader, follower = pair.split('-')
            direction = f"{follower} leads {leader}"
        else:
            direction = "No clear lead-lag relationship"
        
        print(f"{pair}: Lag = {data['lag']} minutes, Direction: {direction}")

# %%
