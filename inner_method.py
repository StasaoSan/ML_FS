import numpy as np
from sklearn.ensemble import RandomForestClassifier


def rf_embedded_selection(X, y, num_features=30, n_estimators=100, random_state=42, max_depth=None):
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, max_depth=max_depth)
    rf.fit(X, y)
    importances = rf.feature_importances_
    sorted_indices = np.argsort(importances)[::-1]
    selected_indices = sorted_indices[:num_features]
    mask = np.zeros(X.shape[1], dtype=bool)
    mask[selected_indices] = True
    return mask, selected_indices
