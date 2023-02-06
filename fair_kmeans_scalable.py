# © 2021, Universität Bern, Chair of Quantitative Methods, Manuel Kammermann, Philipp Baumann

import gurobipy as gb
import multiprocessing
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import kmeans_plusplus
from sklearn.cluster import MiniBatchKMeans
import time


def create_minibatch_cluster(X, colors, mb_clusters, random_state=42):
    """Create representatives using mini-batch k-means

    Args:
        X (np.array): feature vectors of objects
        colors (np.array): colors of objects (0=red; 1=blue)
        mb_clusters (int): number of mini-batch k-means sub-clusters
        random_state (int, RandomState instance): random state

    Returns:
        X_ (np.array): feature vectors of representatives
        num_blue (np.array): number of blue points represented by representative
        num_red (np.array): number of blue points represented by representative
        labels (np.array): current cluster assignments of representatives

    """

    # Initialize mini-batch k-means algorithm
    mb_kmeans = MiniBatchKMeans(n_clusters=mb_clusters, random_state=random_state,
                                batch_size=256*multiprocessing.cpu_count())

    mb_kmeans.fit(X)

    # Store assigment of objects to mini-batch clusters
    labels = mb_kmeans.labels_

    # Store unique mini-batch clusters
    minibatch_cluster, indices = np.unique(labels, return_index=True)

    # Initialize dictionaries
    num_red = dict()
    num_blue = dict()
    mb_index = dict()

    for cluster in minibatch_cluster:

        points = colors[mb_kmeans.labels_ == cluster]
        reds = sum(points)
        blues = len(points) - sum(points)

        num_red[cluster] = reds
        num_blue[cluster] = blues
        mb_index[cluster] = np.argwhere(mb_kmeans.labels_ == cluster).ravel().tolist()

    # Create new representative using mini-batch cluster center
    X_ = mb_kmeans.cluster_centers_[list(mb_index.keys())]

    return X_, num_red, num_blue, labels


def update_centers(X_, centers, n_clusters, labels, algorithm):
    """Update positions of cluster centers

    Args:
        X_ (np.array): feature vectors of representatives
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
            centers[i] = X_[labels == i, :].mean(axis=0)

        # Compute median of each cluster (k-median)
        if algorithm == 'kmedian':
            centers[i] = np.median(X_[labels == i, :], axis=0)

        # Compute medoid of each cluster (k-medoid)
        if algorithm == 'kmedoid':
            # Sample 50'000 random data points if size of cluster is too big
            x_sample = X_[labels == i, :][np.random.choice(X_[labels == i, :].shape[0], min(X_[labels == i, :].shape[0],
                                                                                            50000), replace=False)]

            # Compute pairwise euclidean distance of each data point in cluster_i
            distances = cdist(x_sample, x_sample, metric='euclidean')

            centers[i] = x_sample[np.argmin(distances.sum(axis=0))]

    return centers


def assign_objects(X_, centers, num_red, num_blue, p, q):
    """Assigns objects to clusters

    Args:
        X_ (np.array): feature vectors of representatives
        centers (np.array): current positions of cluster centers
        num_red (dict): tbd
        num_blue (dict): tbd
        p (int): integer used to compute minimum balance
        q (int): integer used to compute minimum balance

    Returns:
        labels (np.array): cluster labels for objects

    """

    # Compute model input
    k = centers.shape[0]
    index_ = list(num_red.keys())

    distances = pd.DataFrame(cdist(X_, centers), index=index_)
    assignments = {(i, j): distances.loc[i, j] for i in index_ for j in range(k)}
    ratio = min(p, q) / max(p, q)

    # Create model
    m = gb.Model()

    # Add binary decision variables
    y = m.addVars(assignments, obj=assignments, vtype=gb.GRB.BINARY)

    # Add constraints
    m.addConstrs(y.sum(i, '*') == 1 for i in index_)
    m.addConstrs(y.sum('*', j) >= 1 for j in range(k))
    m.addConstrs(gb.quicksum(y[i, j] * num_red[i] for i in index_) >= ratio * gb.quicksum(y[i, j] * num_blue[i] for i in
                                                                                          index_) for j in range(k))
    m.addConstrs(gb.quicksum(y[i, j] * num_blue[i] for i in index_) >= ratio * gb.quicksum(y[i, j] * num_red[i] for i in
                                                                                           index_) for j in range(k))

    # Determine optimal solution
    m.setParam('Outputflag', 0)
    m.optimize()

    # Get labels from optimal assignment
    if m.status != gb.GRB.INFEASIBLE:
        labels = np.array([j for i, j in y.keys() if y[i, j].X > 0.5])

    else:
        labels = np.array([])

    return labels


def get_total_distance(X, X_, centers, labels_, labels, algorithm):
    """Computes total distance between objects and cluster centers

    Args:
        X (np.array): feature vectors of objects
        X_ (np.array): feature vectors of representatives
        centers (np.array): current positions of cluster centers
        labels (np.array): current cluster assignments of objects
        labels_ (np.array): current cluster assignments of representatives
        algorithm (str): clustering algorithm (k-means, k-median etc.)

    Returns:
        dist (float): total distance

    """

    assignment = dict(zip(np.unique(labels).tolist(), labels_))

    if algorithm == 'kmeans':
        dist_ = np.sqrt(((X_ - centers[labels_, :]) ** 2).sum(axis=1).sum())
        dist = np.sqrt(((X - centers[np.vectorize(assignment.get)(labels), :]) ** 2).sum(axis=1).sum())

    if algorithm == 'kmedian':
        dist = sum(np.linalg.norm(X_ - centers[labels_, :], axis=1))
        dist_ = sum(np.linalg.norm(X_ - centers[labels_, :], axis=1))

    if algorithm == 'kcenter':
        dist = np.max(np.abs(X_ - centers[labels_, :]), axis=1)
        dist_ = np.max(np.abs(X_ - centers[labels_, :]), axis=1)

    if algorithm == 'kmedoid':
        # dist = np.sqrt(((X - centers[labels, :]) ** 2).sum(axis=1)).sum()
        dist = sum(np.linalg.norm(X_ - centers[labels_, :], axis=1))
        dist_ = sum(np.linalg.norm(X_ - centers[labels_, :], axis=1))

    return dist, dist_


def get_balance(labels, labels_, colors):
    """ Computes balance of the clustering

    Args:
        labels (np.array): current cluster assignments of objects
        labels_ (np.array): current cluster assignments of representatives
        colors (np.array): colors of objects (0=red; 1=blue)

    Returns:
        balance (float): achieved balance

    """

    assignment = dict(zip(np.unique(labels).tolist(), labels_))

    min_ = 1
    for cluster in np.unique(labels_):
        color_dist = colors[np.vectorize(assignment.get)(labels) == cluster]

        r = sum(color_dist)
        b = len(color_dist) - r

        if r == 0 or b == 0:
            min_ = 0

        else:
            min_ = min(min_, r / b, b / r)

    return min_


def scalable_fair_kmeans(X, n_clusters, colors, p, q, algorithm, mb_clusters, random_state, max_iter=100):
    """Finds partition of X subject to balance constraint

    Args:
        X (np.array): feature vectors of objects
        n_clusters (int): predefined number of clusters
        colors (np.array): colors of objects (0=red; 1=blue)
        p (int): integer used to compute minimum balance
        q (int): integer used to compute minimum balance
        algorithm (str): clustering algorithm
        mb_clusters (int): number of mini-batch subclusters
        random_state (int, RandomState instance): random state
        max_iter (int): maximum number of iterations of fair_kmeans algorithm

    Returns:
        best_labels (np.array): cluster labels of objects
        best_total_distance (float): minimal distance (objective function value)
        total_time (float): total running time
        best_total_balance (float): achieved balance in best solution

    """

    # Initialize start time
    start_time = time.time()

    X_, num_red, num_blue, labels = create_minibatch_cluster(X, colors, mb_clusters, random_state=random_state)

    # Choose initial cluster centers randomly
    # np.random.seed(random_state)
    # center_ids = np.random.choice(np.arange(X.shape[0]), size=n_clusters, replace=False)
    # centers = X[center_ids, :]

    # Choose initial cluster using the k-means++ algorithm
    centers, indices = kmeans_plusplus(X_, n_clusters=n_clusters, random_state=random_state, n_local_trials=1000)
    # print(centers)

    # Assign objects
    labels_ = assign_objects(X_, centers, num_red, num_blue, p, q)

    if labels_.size > 0:

        # Initialize best labels
        best_labels = labels_

        # Update centers
        centers = update_centers(X_, centers, n_clusters, labels_, algorithm)

        # Compute total distance
        best_total_distance, best_total_distance_ = get_total_distance(X, X_, centers, labels_, labels,
                                                                       algorithm)

        # Compute balance
        best_total_balance = get_balance(labels, labels_, colors)
        # balance = 1

        n_iter = 1

        while n_iter < max_iter:

            # Assign objects
            labels_ = assign_objects(X_, centers, num_red, num_blue, p, q)

            # Update centers
            centers = update_centers(X_, centers, n_clusters, labels_, algorithm)

            # Compute total distance
            total_distance, total_distance_ = get_total_distance(X, X_, centers, labels_, labels, algorithm)

            # Compute balance
            # balance = get_balance(labels, labels_, colors)

            # Check stopping criterion
            if total_distance_ >= best_total_distance_:
                break

            else:
                # Update best labels and best total distance
                best_labels = labels_
                best_total_distance_ = total_distance_
                best_total_distance = total_distance

                # Increase iteration counter
                n_iter += 1

        total_time = time.time() - start_time
        print('K-Means cost representative: ' + str(best_total_distance_))
        print('K-Means cost overall: ' + str(best_total_distance))
        print('Total running time: ' + str(total_time))

    else:
        best_labels = []
        best_total_distance = 'infeasible'
        total_time = 'infeasible'
        best_total_balance = 'infeasible'
        print('Model is infeasible')

    return best_labels, best_total_distance, total_time, best_total_balance
