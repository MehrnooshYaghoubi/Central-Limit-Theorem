import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def plot_sample_means(scale, n_samples, sample_size):
    """
    Plots the distribution of sample means from an exponential distribution.

    Args:
        scale (float): The scale parameter of the exponential distribution.
        n_samples (int): The number of samples to draw.
        sample_size (int): The size of each sample.
    """

    # Generate data and calculate sample means
    data = np.random.exponential(scale=scale, size=n_samples * sample_size)
    means = [np.mean(data[i * sample_size:(i + 1) * sample_size]) for i in range(n_samples)]

    mean = np.mean(means)
    std = np.std(means)

    plt.figure(figsize=(12, 8))

    plt.hist(means, bins=50, density=True, alpha=0.6, color="#4C72B0", edgecolor='black', label="Sample Means")

    xfit = np.linspace(min(means), max(means), 100)
    yfit = norm.pdf(xfit, mean, std)
    plt.plot(xfit, yfit, 'r-', lw=2, label=f"Normal Fit (μ={mean:.2f}, σ={std:.2f})")

    plt.axvline(mean, color='orange', linestyle='--', linewidth=2, label=f"Mean (μ={mean:.2f})")

    plt.annotate(f"μ = {mean:.2f}\nσ = {std:.2f}", 
                 xy=(mean, max(yfit) * 0.8), 
                 xytext=(mean + std, max(yfit) * 0.9),
                 arrowprops=dict(facecolor='black', arrowstyle="->"),
                 fontsize=12, color='darkred')

    plt.title('Distribution of Sample Means', fontsize=16, fontweight='bold')
    plt.xlabel('Sample Mean', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example usage
    plot_sample_means(scale=2.0, n_samples=1000, sample_size=30)