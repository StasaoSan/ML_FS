import numpy as np
from sklearn.feature_selection import f_classif, SelectKBest, SelectFromModel, SequentialFeatureSelector, \
    VarianceThreshold
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression


def remove_constant_features(X, feature_names=None, selector=None):
    if selector is None:
        selector = VarianceThreshold(threshold=0)
        X_new = selector.fit_transform(X)
        if feature_names is not None:
            support_mask = selector.get_support()
            updated_feature_names = feature_names[support_mask]
            return X_new, updated_feature_names, selector
        else:
            return X_new, None, selector
    else:
        X_new = selector.transform(X)
        return X_new, feature_names, selector


def filter_lib(X_train, y_train, X_val, y_val, all_feature_names, classifiers, evaluate_classifiers, k=30):
    X_train, all_feature_names, vt_selector = remove_constant_features(X_train, all_feature_names)
    X_val, _, _ = remove_constant_features(X_val, all_feature_names, vt_selector)

    selector = SelectKBest(f_classif, k=k)
    X_train_new = selector.fit_transform(X_train, y_train)
    selected_indices = selector.get_support(indices=True)
    top_features = all_feature_names[selected_indices]

    print("\nTop 30 features wia filter method (f_classif):")
    for feature in top_features[:30]:
        print(feature)

    if isinstance(X_val, np.ndarray):
        X_val_new = X_val[:, selected_indices]
    else:
        X_val_new = X_val.toarray()[:, selected_indices]

    accuracies_after_filter = evaluate_classifiers(classifiers, X_train_new, y_train, X_val_new, y_val)
    return accuracies_after_filter, selected_indices, top_features, X_train_new, X_val_new


def inner_lib(X_train, y_train, X_val, y_val, all_feature_names, classifiers, evaluate_classifiers, threshold='mean'):
    X_train, all_feature_names, vt_selector = remove_constant_features(X_train, all_feature_names)
    X_val, _, _ = remove_constant_features(X_val, all_feature_names, vt_selector)
    lsvc = LinearSVC(C=0.5, penalty="l1", random_state=42, max_iter=1000)
    model = SelectFromModel(lsvc, threshold=threshold)
    X_train_new = model.fit_transform(X_train, y_train)
    selected_indices = model.get_support(indices=True)
    top_features = all_feature_names[selected_indices]

    print("\nTop 30 features wia embedded method (LinearSVC + L1):")
    for f in top_features[:30]:
        print(f)

    X_val_new = X_val[:, selected_indices]
    accuracies_after_inner = evaluate_classifiers(classifiers, X_train_new, y_train, X_val_new, y_val)
    return accuracies_after_inner, selected_indices, top_features, X_train_new, X_val_new


def wrapper_lib(X_train, y_train, X_val, y_val, all_feature_names, classifiers, evaluate_classifiers, n_features=30):
    X_train, all_feature_names, vt_selector = remove_constant_features(X_train, all_feature_names)
    X_val, _, _ = remove_constant_features(X_val, all_feature_names, vt_selector)

    base_estimator = LogisticRegression(max_iter=1000, random_state=42)
    sfs = SequentialFeatureSelector(base_estimator, n_features_to_select=n_features, direction='forward', n_jobs=-1)
    X_train_new = sfs.fit_transform(X_train, y_train)
    selected_indices = sfs.get_support(indices=True)
    top_features = all_feature_names[selected_indices]

    print("\nTop 30 features wia wrapper method (SFS):")
    for f in top_features[:30]:
        print(f)

    X_val_new = X_val[:, selected_indices]
    accuracies_after_wrapper = evaluate_classifiers(classifiers, X_train_new, y_train, X_val_new, y_val)
    return accuracies_after_wrapper, selected_indices, top_features, X_train_new, X_val_new
