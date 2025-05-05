#%%
import numpy as np
import matplotlib.pyplot as plt

def penalized_least_squares(y, lambda_):
    """
    Perform Penalized Least Squares smoothing on a 1D array.
    
    Parameters:
        y (array-like): Input data to be smoothed.
        lambda_ (float): Regularization parameter controlling smoothness.
        
    Returns:
        smoothed (array-like): Smoothed data.
    """
    n = len(y)
    D = np.diff(np.eye(n), 2)  # Second-order difference matrix
    penalty_matrix = lambda_ * D.T @ D
    smoothed = np.linalg.solve(np.eye(n-2) + penalty_matrix, y[2:])
    return np.concatenate(([y[0], y[1]], smoothed))


if __name__ == "__main__":
    # Example: Simulated financial data
    np.random.seed(42)
    x = np.linspace(0, 10, 100)
    y = np.sin(x) + np.random.normal(scale=0.2, size=len(x))  # Noisy data

    lambda_ = 10  # Regularization parameter
    smoothed_y = penalized_least_squares(y, lambda_)

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label="Noisy Data", alpha=0.6)
    plt.plot(x, smoothed_y, label="Smoothed Data", linewidth=2)
    plt.legend()
    plt.title("Penalized Least Squares Smoothing")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.show()