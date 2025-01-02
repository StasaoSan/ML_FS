import numpy as np
from sklearn.base import clone

def manual_rfe(X, y, base_estimator, n_features_to_select=30):
    selected_features = list(range(X.shape[1]))
    ranking = []

    while len(selected_features) > n_features_to_select:
        clf = clone(base_estimator)
        clf.fit(X[:, selected_features], y)

        if hasattr(clf, 'feature_importances_'):
            importances = clf.feature_importances_
        elif hasattr(clf, 'coef_'):
            importances = np.abs(clf.coef_).flatten()
        else:
            raise ValueError("Base classifier did not support feature_importances_ or coef_.")

        least_important_index = np.argmin(importances)
        least_important_feature = selected_features[least_important_index]

        ranking.append(least_important_feature)

        selected_features.pop(least_important_index)

    mask = np.zeros(X.shape[1], dtype=bool)
    mask[selected_features] = True

    clf_final = clone(base_estimator)
    clf_final.fit(X[:, selected_features], y)

    if hasattr(clf_final, 'feature_importances_'):
        final_importances = clf_final.feature_importances_
    elif hasattr(clf_final, 'coef_'):
        final_importances = np.abs(clf_final.coef_).flatten()
    else:
        raise ValueError("Base classifier did not support feature_importances_ or coef_.")

    sorted_indices = np.argsort(final_importances)[::-1]
    sorted_features = [selected_features[i] for i in sorted_indices]
    ranking = sorted_features + ranking

    ranking = np.array(ranking)

    return mask, ranking
