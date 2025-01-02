from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def apply_pca(X, n_components=2, random_state=42):
    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X)
    return X_pca


def apply_tsne(X, n_components=2, random_state=42, perplexity=30):
    tsne = TSNE(n_components=n_components, random_state=random_state, perplexity=perplexity, init='pca')
    X_tsne = tsne.fit_transform(X)
    return X_tsne
