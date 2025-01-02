import matplotlib.pyplot as plt
import numpy as np


def plot_data_with_class(X_2d, y, title="Original classes", xlabel="Dim 1", ylabel="Dim 2"):
    unique_classes = np.unique(y)
    plt.figure(figsize=(6, 5))
    for cls in unique_classes:
        mask = (y == cls)
        cluster_size = np.sum(mask)
        plt.scatter(X_2d[mask, 0], X_2d[mask, 1], label=f"Class {cls} (N={cluster_size})", s=10)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_data_with_cluster(X_2d, labels, title="Clusters", xlabel="Dim 1", ylabel="Dim 2"):
    unique_clusters = np.unique(labels)
    plt.figure(figsize=(6, 5))

    for cluster in unique_clusters:
        mask = (labels == cluster)
        cluster_size = np.sum(mask)
        plt.scatter(X_2d[mask, 0], X_2d[mask, 1], s=10, label=f"Cluster {cluster} (N={cluster_size})")

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.show()
