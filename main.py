import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, adjusted_rand_score, silhouette_score
from sklearn.cluster import KMeans

from build_data import get_processed_data
from filter_method import manual_chi2
from inner_method import rf_embedded_selection
from lib_realizations import filter_lib, inner_lib, wrapper_lib
from wrapper_method import manual_rfe

from dimenshion_reduce import apply_pca, apply_tsne
from plot_graps import plot_data_with_class, plot_data_with_cluster

def evaluate_classifiers(classifiers, X_train, y_train, X_val, y_val):
    accuracies = []
    for name, clf in classifiers:
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        accuracies.append(acc)
        print(f"{name}: {acc:.4f}")
    return accuracies

def main():
    print("1. Load data...")
    X_train, y_train, X_val, y_val, all_feature_names = get_processed_data("Corona_NLP_test.csv")

    classifiers = [
        ("LogisticRegression", LogisticRegression(max_iter=1000, random_state=42)),
        ("RandomForest", RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)),
        ("SVC", SVC(random_state=42))
    ]

    print("\n=== Accuracy without FS ===")
    accuracies_before_fs = evaluate_classifiers(classifiers, X_train, y_train, X_val, y_val)

    fs_accuracies = {"No_FS": accuracies_before_fs}

    print("\n2. Filter method (chi^2)...")
    chi2_scores = manual_chi2(X_train, y_train)
    chi2_sorted_indices = np.argsort(chi2_scores)[::-1]

    top_k_chi2 = 30
    selected_indices_chi2 = chi2_sorted_indices[:top_k_chi2]

    X_train_chi2 = X_train[:, selected_indices_chi2]
    X_val_chi2 = X_val[:, selected_indices_chi2]

    top_30_chi2_features = all_feature_names[selected_indices_chi2]

    print("\nTop 30 wia chi^2:")
    for f in top_30_chi2_features:
        print(f)

    print("\n=== Accuracy after chi^2 FS ===")
    accuracies_chi2 = evaluate_classifiers(classifiers, X_train_chi2, y_train, X_val_chi2, y_val)
    fs_accuracies["Chi2"] = accuracies_chi2

    print("\n3. Embedded method (Random Forest FS)...")
    mask_embedded, ranking_embedded = rf_embedded_selection(
        X_train, y_train, num_features=30, n_estimators=100, random_state=42
    )

    X_train_embedded = X_train[:, mask_embedded]
    X_val_embedded = X_val[:, mask_embedded]

    top_30_embedded_indices = ranking_embedded[:30]
    top_30_embedded_features = all_feature_names[top_30_embedded_indices]
    print("\nTop 30 wia embedded method (RF):")
    for f in top_30_embedded_features:
        print(f)

    print("\n=== Accuracy afrer embedded method (RF) ===")
    accuracies_embedded = evaluate_classifiers(classifiers, X_train_embedded, y_train, X_val_embedded, y_val)
    fs_accuracies["Embedded_RF"] = accuracies_embedded

    print("\n4. Wrapper method (Custom RFE)...")
    base_clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    n_features_to_select = 30

    mask_wr, ranking_wr = manual_rfe(
        X_train, y_train,
        base_estimator=base_clf,
        n_features_to_select=n_features_to_select
    )

    X_train_wr = X_train[:, mask_wr]
    X_val_wr = X_val[:, mask_wr]

    top_30_wr_features = all_feature_names[ranking_wr[:30]]
    print("\nTop 30 wia wrapper method (RFE):")
    for f in top_30_wr_features:
        print(f)

    print("\n=== Accuracy after custom_RFE ===")
    accuracies_wr = evaluate_classifiers(classifiers, X_train_wr, y_train, X_val_wr, y_val)
    fs_accuracies["Wrapper_RFE"] = accuracies_wr


    print("\n=== Lib FS methods ===")
    accuracies_filter, filter_indices, filter_features, X_train_filter, X_val_filter = filter_lib(
        X_train, y_train, X_val, y_val, all_feature_names, classifiers, evaluate_classifiers, k=30
    )
    fs_accuracies["Filter_f_classif_lib"] = accuracies_filter


    accuracies_inner, inner_indices, inner_features, X_train_inner, X_val_inner = inner_lib(
        X_train, y_train, X_val, y_val, all_feature_names, classifiers, evaluate_classifiers, threshold='mean'
    )
    fs_accuracies["Inner_L1SVC_lib"] = accuracies_inner


    accuracies_wrapper, wrapper_indices, wrapper_features, X_train_wrapper, X_val_wrapper = wrapper_lib(
        X_train, y_train, X_val, y_val, all_feature_names, classifiers, evaluate_classifiers, n_features=30
    )
    fs_accuracies["Wrapper_SFS_lib"] = accuracies_wrapper


    print("\nChoose best FS method...")
    average_accuracies = {method: np.mean(acc) for method, acc in fs_accuracies.items()}
    for method, avg_acc in average_accuracies.items():
        print(f"{method}: average accuracy = {avg_acc:.4f}")
    best_fs_method = max(average_accuracies, key=average_accuracies.get)
    print(f"\nBest FS method: {best_fs_method} with accuracy {average_accuracies[best_fs_method]:.4f}")

    if best_fs_method == "No_FS":
        X_train_best_fs = X_train
    elif best_fs_method == "Chi2_manual":
        X_train_best_fs = X_train_chi2
    elif best_fs_method == "Embedded_RF_manual":
        X_train_best_fs = X_train_embedded
    elif best_fs_method == "Wrapper_RFE_manual":
        X_train_best_fs = X_train_wr
    elif best_fs_method == "Filter_f_classif_lib":
        X_train_best_fs = X_train_filter
    elif best_fs_method == "Inner_L1SVC_lib":
        X_train_best_fs = X_train_inner
    elif best_fs_method == "Wrapper_SFS_lib":
        X_train_best_fs = X_train_wrapper
    else:
        raise ValueError("Unexpecded FS method")


    print("\nClusterization...")
    kmeans_before = KMeans(n_clusters=5, random_state=42)
    kmeans_before.fit(X_train)
    labels_before = kmeans_before.labels_

    kmeans_after = KMeans(n_clusters=5, random_state=42)
    kmeans_after.fit(X_train_best_fs)
    labels_after = kmeans_after.labels_

    ari_before = adjusted_rand_score(y_train, labels_before)
    sil_before = silhouette_score(X_train, labels_before)
    ari_after = adjusted_rand_score(y_train, labels_after)
    sil_after = silhouette_score(X_train_best_fs, labels_after)

    print("\nClusterization quality:")
    print(f"Before FS: ARI={ari_before:.4f}, Silhouette={sil_before:.4f}")
    print(f"After FS ({best_fs_method}): ARI={ari_after:.4f}, Silhouette={sil_after:.4f}")


    X_train_pca = apply_pca(X_train)
    X_train_best_fs_pca = apply_pca(X_train_best_fs)

    plot_data_with_class(X_train_pca, y_train, title=f"Original classes (PCA, BEFORE FS)")
    plot_data_with_class(X_train_best_fs_pca, y_train, title=f"Original classes (PCA, AFTER FS: {best_fs_method})")

    plot_data_with_cluster(X_train_pca, labels_before, title="Clusters (PCA, BEFORE FS)")
    plot_data_with_cluster(X_train_best_fs_pca, labels_after,
                           title=f"Clusters (PCA, AFTER FS: {best_fs_method})")


    X_train_tsne = apply_tsne(X_train)
    X_train_best_fs_tsne = apply_tsne(X_train_best_fs)

    plot_data_with_class(X_train_tsne, y_train, title="Original classes (t-SNE, BEFORE FS)")
    plot_data_with_class(X_train_best_fs_tsne, y_train, title=f"Original classes (t-SNE, AFTER FS: {best_fs_method})")

    plot_data_with_cluster(X_train_tsne, labels_before, title="Clusters (t-SNE, BEFORE FS)")
    plot_data_with_cluster(X_train_best_fs_tsne, labels_after,
                           title=f"Clusters (t-SNE, AFTER FS: {best_fs_method})")


if __name__ == "__main__":
    main()
