import numpy as np


def manual_chi2(X, y):
    n_samples, n_features = X.shape
    classes = np.unique(y)
    n_classes = len(classes)

    class_counts = np.array([(y == c).sum() for c in classes])
    counts = np.zeros((n_classes, n_features), dtype=int)

    for i in range(n_samples):
        row_features = np.where(X[i] != 0)[0]
        class_idx = np.searchsorted(classes, y[i])
        for f in row_features:
            counts[class_idx, f] += 1

    total_presence = counts.sum(axis=0)
    total_absence = n_samples - total_presence
    chi2_scores = np.zeros(n_features, dtype=float)

    for f in range(n_features):
        observed_presence = counts[:, f]
        observed_absence = class_counts - observed_presence
        if total_presence[f] == 0 or total_presence[f] == n_samples:
            chi2_scores[f] = 0
            continue

        expected_presence = class_counts * (total_presence[f] / n_samples)
        expected_absence = class_counts * (total_absence[f] / n_samples)
        chi2_val = 0.0

        for c_idx in range(n_classes):
            if expected_presence[c_idx] > 0:
                chi2_val += ((observed_presence[c_idx] - expected_presence[c_idx]) ** 2) / expected_presence[c_idx]
            if expected_absence[c_idx] > 0:
                chi2_val += ((observed_absence[c_idx] - expected_absence[c_idx]) ** 2) / expected_absence[c_idx]

        chi2_scores[f] = chi2_val

    return chi2_scores
