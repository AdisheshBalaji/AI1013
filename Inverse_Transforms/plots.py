import distributions
import matplotlib.pyplot as plt
import numpy as np

n_samples = 10000

def plot_bernoulli(p):
    data_itt = [distributions.bernoulli(p) for _ in range(n_samples)]
    data_py = np.random.binomial(1, p, n_samples)

    fig, ax = plt.subplots(1, 2, figsize=(16, 4))
    fig.suptitle(f"Bernoulli RV with probability {p}")

    ax[0].hist(data_itt, bins=[-0.5, 0.5, 1.5], label="Inverse Transform", edgecolor="black")
    ax[0].set_xlabel("Values")
    ax[0].set_ylabel("Frequency")
    ax[0].legend()
    ax[0].set_xticks([0, 1])
    ax[0].grid(True)

    ax[1].hist(data_py, bins=[-0.5, 0.5, 1.5], label="True Distribution", edgecolor="black")
    ax[1].set_xlabel("Values")
    ax[1].set_ylabel("Frequency")
    ax[1].legend()
    ax[1].set_xticks([0, 1])
    ax[1].grid(True)

    plt.savefig("plots/Bernoulli.png")
    plt.close()


def plot_geometric(p):
    data_itt = [distributions.geometric(p) for _ in range(n_samples)]
    data_py = np.random.geometric(p, size=n_samples)

    fig, ax = plt.subplots(1, 2, figsize=(16, 4))
    fig.suptitle(f"Geometric RV with probability {p}")

    values_py, counts_py = np.unique(data_py, return_counts=True)
    probs_py = counts_py / n_samples

    values, counts = np.unique(data_itt, return_counts=True)
    probs = counts / n_samples

    ax[0].plot(values_py, probs_py, marker="o", color="green", label="True Distribution")
    ax[0].set_xlabel("Values")
    ax[0].set_ylabel("Frequency")
    ax[0].legend()
    ax[0].grid(True)

    ax[1].plot(values, probs, marker="o", color="blue", label="Inverse Transform")
    ax[1].set_xlabel("Values")
    ax[1].set_ylabel("Frequency")
    ax[1].legend()
    ax[1].grid(True)

    plt.savefig("plots/Geometric.png")
    plt.close()


def plot_exponential(t):
    data_itt = [distributions.exponential(t) for _ in range(n_samples)]

    fig, ax = plt.subplots()
    x = np.linspace(0, np.max(data_itt), 10000)
    pdf = t * np.exp(-t * x)

    ax.hist(data_itt, bins=50, density=True, alpha=0.5, label="Inverse Transform")
    ax.plot(x, pdf, 'r-', lw=2, label="True Distribution")
    ax.set_xlabel("x")
    ax.set_ylabel("Probability Density")
    ax.set_title(f"Exponential RV with parameter {t}")
    ax.legend()

    plt.savefig("plots/Exponential.png")
    plt.close()


def plot_rayleigh():
    data_itt = [distributions.rayleigh() for _ in range(n_samples)]

    fig, ax = plt.subplots()
    x = np.linspace(0, np.max(data_itt), 10000)
    pdf = x * np.exp(-x**2 / 2)

    ax.hist(data_itt, bins=50, density=True, alpha=0.5, label="Inverse Transform")
    ax.plot(x, pdf, "r-", lw=2, label="True Distribution")
    ax.set_xlabel("x")
    ax.set_ylabel("Probability Density")
    ax.set_title("Rayleigh RV")
    ax.legend()

    plt.savefig("plots/Rayleigh.png")
    plt.close()


def plot_gaussian(mu, var):
    data_itt = [distributions.gaussian(mu, var) for _ in range(n_samples)]

    fig, ax = plt.subplots()
    x = np.linspace(-np.max(data_itt), np.max(data_itt), 10000)
    pdf = (1 / np.sqrt(2 * np.pi * var)) * np.exp(-((x - mu) ** 2) / (2 * var))

    ax.hist(data_itt, bins=50, density=True, alpha=0.5, label="Inverse Transform")
    ax.plot(x, pdf, "r-", lw=2, label="True Distribution")
    ax.set_xlabel("x")
    ax.set_ylabel("Probability Density")
    ax.set_title(f"Gaussian RV with mean {mu} and variance {var}")
    ax.legend()

    plt.savefig("plots/Gaussian.png")
    plt.close()


# Generate plots
plot_bernoulli(0.3)
plot_geometric(0.2)
plot_exponential(2)
plot_rayleigh()
plot_gaussian(0, 1)
