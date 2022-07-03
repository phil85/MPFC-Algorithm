from configparser import ConfigParser
from fair_kmeans import fair_kmeans
# from fair_kmeans_scalable import scalable_fair_kmeans
import numpy as np
from preprocessing import load_data
from fractions import Fraction
import matplotlib.pyplot as plt

# %% Determine control parameters
data = 'illustrative_example'
algorithm = 'kmeans'
n_clusters = 4


# %% Read config parameters
config = ConfigParser()
config.read('data_config.ini')


# %% Define parameters
dataset = config.get(data, 'file')
attributes = config.get(data, 'attributes').split(',')
sensitive_attribute = [config.get(data, 'sensitive_attribute')]
minimum_balance = Fraction(config.get(data, 'minimum_balance')).limit_denominator()
dataset_balance = Fraction(config.get(data, 'dataset_balance')).limit_denominator()


# %% Import dataset
df = load_data(dataset, attributes, sensitive_attribute, normalize=False, standardize=False, sample=False)
X = df.iloc[:, :-1].values
colors = df.iloc[:, -1].values


# %% Run mathematical programming-based fair k-clustering algorithm (MPFC)
p = minimum_balance.numerator
q = minimum_balance.denominator

labels, total_distance, total_time, balance = fair_kmeans(X, n_clusters=n_clusters, colors=colors, p=p, q=q,
                                                          algorithm=algorithm, random_state=42)


# %% Visualize data set
red = colors == 0
blue = colors == 1
plt.scatter(X[red, 0], X[red, 1], color='red')
plt.scatter(X[blue, 0], X[blue, 1], color='blue')
plt.show()

# Plot solution only if model is feasible
if len(labels) > 0:

    # Compute cluster centers
    centers = np.zeros((n_clusters, X.shape[1]))
    for i in range(n_clusters):
        centers[i] = X[labels == i, :].mean(axis=0)

    # Visualize solution
    plt.scatter(X[red, 0], X[red, 1], color='red')
    plt.scatter(X[blue, 0], X[blue, 1], color='blue')
    plt.scatter(centers[:, 0], centers[:, 1], marker='x', color='k', zorder=3)
    for i in range(X.shape[0]):
        plt.plot([X[i, 0], centers[labels[i], 0]], [X[i, 1], centers[labels[i], 1]], color='gray', alpha=0.25)
    plt.show()
