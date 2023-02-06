# © 2021, Universität Bern, Chair of Quantitative Methods, Manuel Kammermann, Philipp Baumann

import gurobipy as gb
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import kmeans_plusplus
import time


def update_centers(X, centers, n_clusters, labels, algorithm):
    """Update positions of cluster centers

    Args:
        X (np.array): feature vectors of objects
        centers (np.array): current positions of cluster centers
        n_clusters (int): predefined number of clusters
        labels (np.array): current cluster assignments of objects
        algorithm (str): clustering algorithm (k-means, k-median etc.)

    Returns:
        centers (np.array): the updated positions of cluster centers

    """

    for i in range(n_clusters):

        # Compute mean of each cluster (k-means or k-center)
        if algorithm == 'kmeans' or algorithm == 'kcenter':
            centers[i] = X[labels == i, :].mean(axis=0)

        # Compute median of each cluster (k-median)
        if algorithm == 'kmedian':
            centers[i] = np.median(X[labels == i, :], axis=0)

        # Compute medoid of each cluster (k-medoid): If number of objects exceed 50,000 then randomly sample
        # 50,000 objects from the cluster
        if algorithm == 'kmedoid':

            # Sample 50'000 random objects if size of cluster is too big
            x_sample = X[labels == i, :][np.random.choice(X[labels == i, :].shape[0],
                                                          min(X[labels == i, :].shape[0], 50000), replace=False)]

            # Compute pairwise euclidean distance of each data point in cluster
            distances = cdist(x_sample, x_sample, metric='euclidean')

            centers[i] = x_sample[np.argmin(distances.sum(axis=0))]

    return centers


def assign_objects(X, centers, colors, p, q):
    """Assigns objects to clusters

    Args:
        X (np.array): feature vectors of objects
        centers (np.array): current positions of cluster centers
        colors (np.array): colors of objects (0=red; 1=blue)
        p (int): integer used to compute minimum balance with balance = min(p/q, q/p)
        q (int): integer used to compute minimum balance with balance = min(p/q, q/p)

    Returns:
        labels (np.array): cluster labels for objects

    """

    # Compute model input
    n = X.shape[0]
    k = centers.shape[0]
    distances = cdist(X, centers)
    assignments = {(i, j): distances[i, j] for i in range(n) for j in range(k)}
    red = np.where(colors == 0)[0]
    blue = np.where(colors == 1)[0]
    ratio = min(p, q) / max(p, q)

    # Create model
    m = gb.Model()

    # Add binary decision variables
    y = m.addVars(assignments, obj=assignments, vtype=gb.GRB.BINARY)

    # Add constraints
    m.addConstrs(y.sum(i, '*') == 1 for i in range(n))
    m.addConstrs(y.sum('*', j) >= 1 for j in range(k))
    m.addConstrs(gb.quicksum(y[i, j] for i in red) >= ratio * gb.quicksum(y[i, j] for i in blue) for j in range(k))
    m.addConstrs(gb.quicksum(y[i, j] for i in blue) >= ratio * gb.quicksum(y[i, j] for i in red) for j in range(k))

    # Determine optimal solution
    m.setParam('Outputflag', 0)
    m.optimize()

    # Get labels from optimal assignment
    if m.status != gb.GRB.INFEASIBLE:
        labels = np.array([j for i, j in y.keys() if y[i, j].X > 0.5])

    else:
        labels = np.array([])

    return labels


def get_total_distance(X, centers, labels, algorithm):
    """Computes total distance between objects and cluster centers

    Args:
        X (np.array): feature vectors of objects
        centers (np.array): current positions of cluster centers
        labels (np.array): current cluster assignments of objects
        algorithm (str): clustering algorithm (kmeans, k-median etc.)

    Returns:
        dist (float): total distance

    """

    # Determine clustering cost for different objective functions
    if algorithm == 'kmeans':
        dist = np.sqrt((((X - centers[labels, :]) ** 2).sum(axis=1)).sum())

    elif algorithm == 'kmedian':
        dist = sum(np.linalg.norm(X - centers[labels, :], axis=1))

    elif algorithm == 'kcenter':
        dist = np.max(np.abs(X - centers[labels, :]), axis=1)

    elif algorithm == 'kmedoid':
        # dist = np.sqrt(((X - centers[labels, :]) ** 2).sum(axis=1)).sum()
        dist = sum(np.linalg.norm(X - centers[labels, :], axis=1))

    return dist


def get_balance(labels, colors):
    """ Computes balance of the clustering

    Args:
        labels (np.array): current cluster assignments of objects
        colors (np.array): colors of objects (0=red; 1=blue)

    Returns:
        min_ (float): achieved balance

    """

    # Initialize balance with the highest possible value of 1
    min_ = 1

    # Determine best achieved balance
    for cluster in np.unique(labels):

        # Determine number of red and blue objects within the cluster
        r = sum(colors[labels == cluster])
        b = len(colors[labels == cluster]) - r

        # Compute balance
        if r == 0 or b == 0:
            min_ = 0

        else:
            min_ = min(min_, r / b, b / r)

    return min_


def fair_kmeans(X, n_clusters, colors, p, q, algorithm, random_state, max_iter=100):
    """Finds partition of X subject to balance constraint

    Args:
        X (np.array): feature vectors of objects
        n_clusters (int): predefined number of clusters
        colors (np.array): colors of objects (0=red; 1=blue)
        p (int): integer used to compute minimum balance
        q (int): integer used to compute minimum balance
        algorithm (str): clustering algorithm
        random_state (int, RandomState instance): random state
        max_iter (int): maximum number of iterations of fair_kmeans algorithm

    Returns:
        best_labels (np.array): cluster labels of objects
        best_total_distance (float): minimal distance (objective function value)
        total_time (float): total running time
        best_total_balance (float): achieved balance in best solution

    """

    # Initialize start time and iteration step
    start_time = time.time()
    n_iter = 1

    # Choose initial cluster centers randomly
    # np.random.seed(random_state)
    # center_ids = np.random.choice(np.arange(X.shape[0]), size=n_clusters, replace=False)
    # centers = X[center_ids, :]

    # Choose initial cluster using the k-means++ algorithm
    centers, indices = kmeans_plusplus(X, n_clusters=n_clusters, random_state=random_state, n_local_trials=1000)

    # Assign objects
    labels = assign_objects(X, centers, colors, p, q)

    if labels.size > 0:

        # Initialize best labels
        best_labels = labels

        # Update centers
        centers = update_centers(X, centers, n_clusters, labels, algorithm)

        # Compute total distance
        best_total_distance = get_total_distance(X, centers, labels, algorithm)

        # Compute balance
        best_total_balance = get_balance(labels, colors)

        while n_iter < max_iter:

            # Assign objects
            labels = assign_objects(X, centers, colors, p, q)

            # Update centers
            centers = update_centers(X, centers, n_clusters, labels, algorithm)

            # Compute total distance
            total_distance = get_total_distance(X, centers, labels, algorithm)

            # Compute balance
            balance = get_balance(labels, colors)

            # Check stopping criterion
            if total_distance >= best_total_distance:
                break

            else:
                # Update best labels and best total distance
                best_labels = labels
                best_total_distance = total_distance
                best_total_balance = balance

                # Increase iteration counter
                n_iter += 1

        total_time = time.time() - start_time
        print('K-Means cost: ' + str(best_total_distance))
        print('Total running time: ' + str(total_time))

    else:
        best_labels = []
        best_total_distance = 'infeasible'
        total_time = 'infeasible'
        best_total_balance = 'infeasible'
        print('Model is infeasible')

    return best_labels, best_total_distance, total_time, best_total_balance
