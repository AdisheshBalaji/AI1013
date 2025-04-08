import distributions
import matplotlib.pyplot as plt
import numpy as np


n_samples = 10000

def plot_bernoulli(p):
    data_itt = []
    data_py = np.random.binomial(1, p, n_samples)
    for _ in range(n_samples):
        bernoulli = distributions.bernoulli(p)
        data_itt.append(bernoulli)
    fig, ax = plt.subplots(1, 2, figsize = (16, 4))
    fig.suptitle(f"Bernoulli RV with probability {p}")

    ax[0].hist(data_itt, bins = [-0.5, 0.5, 1.5], label = "Inverse Transform", edgecolor = "black")
    ax[0].set_xlabel("Values")
    ax[0].set_ylabel("Frequency")
    ax[0].legend()
    ax[0].set_xticks([0, 1])
    ax[0].set_xticklabels(["0", "1"])
    ax[0].grid(True)
    
    ax[1].hist(data_py, bins = [-0.5, 0.5, 1.5], label = "True Distribution", edgecolor = "black")
    ax[1].set_xlabel("Values")
    ax[1].set_ylabel("Frequency")
    ax[1].legend()
    ax[1].set_xticks([0, 1])
    ax[1].set_xticklabels(["0", "1"])
    ax[1].grid(True)

    plt.savefig("plots/Bernoulli.png")

def plot_geometric(p):
    data_itt = []
    data_py = np.random.geometric(p, size = n_samples)
    for _ in range(n_samples):
        geometric = distributions.geometric(p)
        data_itt.append(geometric)
    fig, ax = plt.subplots(1, 2, figsize = (16, 4))
    fig.suptitle(f"Geometric RV with probability {p}")
    
    values_py, counts_py = np.unique(data_py, return_counts=True)
    probs_py = counts_py / n_samples

    values, counts = np.unique(data_itt, return_counts=True)
    probs = counts / n_samples

    ax[0].plot(values_py, probs_py, marker = "o", color = "green", label = "True Distribution")
    ax[0].set_xlabel("Values")
    ax[0].set_ylabel("Frequency")
    ax[0].legend()
    ax[0].grid(True)

    ax[1].plot(values, probs, marker = "o", color = "blue", label = "Inverse Transform")
    ax[1].set_xlabel("Values")
    ax[1].set_ylabel("Frequency")
    ax[1].legend()
    ax[1].grid(True)

    plt.savefig("plots/Geometric.png")

def plot_exponential(t):
    data_itt = []
    for _ in range(n_samples):
        data_itt.append(distributions.exponential(t))


    x = np.linspace(0, np.max(data_itt), 10000)
    pdf = t * np.exp(-t * x)


    plt.hist(data_itt, bins=50, density=True, alpha=0.5, label='Inverse Transform')
    plt.plot(x, pdf, 'r-', lw=2, label='True Distribution')
    plt.xlabel("x")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.title(f"Exponential RV with parameter {t}")
    plt.savefig("plots/Exponential.png")
    

def plot_rayleigh():
    data_itt = []
    for _ in range(n_samples):
        data_itt.append(distributions.rayleigh())

    x = np.linspace(0, np.max(data_itt), 10000)
    pdf = x * np.exp(-x**2/2)

    plt.hist(data_itt, bins = 50, density=True, alpha = 0.5, label = "Inverse Transform")
    plt.plot(x, pdf, "r-", lw=2, label = "True Distribution")
    plt.xlabel("x")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.savefig("plots/Rayleigh.png")
    plt.title("Rayleigh RV")

def plot_gaussian(mu, var):
    data_itt = []
    for _ in range(n_samples):
        data_itt.append(distributions.gaussian(mu, var))
    
    x = np.linspace(-np.max(data_itt), np.max(data_itt), 10000)
    pdf = 1/np.sqrt(2*np.pi*var)*np.exp(-(x - mu)**2/2*var)

    plt.hist(data_itt, bins = 50, density=True, alpha = 0.5, label = "Inverse Transform")
    plt.plot(x, pdf, "r-", lw=2, label = "True Distribution")
    plt.xlabel("x")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.title(f"Gaussian RV with mean {mu} and variance {var}")
    plt.savefig("plots/Gaussian.png")
    

