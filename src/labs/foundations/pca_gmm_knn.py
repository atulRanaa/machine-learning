"""
Lab: Foundations - PCA from Scratch, GMM with EM, and K-Nearest Neighbors
==========================================================================
These implementations show the internal mechanics of classical unsupervised
and supervised algorithms, built from first principles using NumPy.
"""
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt


# =============================================================================
# PCA FROM SCRATCH
# =============================================================================
class PCAFromScratch:
    """Principal Component Analysis implemented from eigendecomposition."""

    def __init__(self, n_components: int):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.eigenvalues = None

    def fit(self, X: np.ndarray):
        self.mean = X.mean(axis=0)
        X_centered = X - self.mean
        cov_matrix = np.cov(X_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        sorted_idx = np.argsort(eigenvalues)[::-1]
        self.eigenvalues = eigenvalues[sorted_idx]
        self.components = eigenvectors[:, sorted_idx[:self.n_components]]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) @ self.components

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

    def explained_variance_ratio(self) -> np.ndarray:
        return self.eigenvalues[:self.n_components] / self.eigenvalues.sum()


# =============================================================================
# GAUSSIAN MIXTURE MODEL (EM ALGORITHM)
# =============================================================================
class GMMFromScratch:
    """Gaussian Mixture Model fitted via the EM algorithm."""

    def __init__(self, k: int = 3, max_iter: int = 100, tol: float = 1e-6):
        self.k = k
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X: np.ndarray) -> np.ndarray:
        n, d = X.shape
        rng = np.random.default_rng(42)

        # Initialize
        self.means = X[rng.choice(n, self.k, replace=False)]
        self.covs = [np.eye(d) for _ in range(self.k)]
        self.weights = np.ones(self.k) / self.k

        prev_ll = -np.inf
        for iteration in range(self.max_iter):
            # E-step
            resp = np.zeros((n, self.k))
            for j in range(self.k):
                resp[:, j] = self.weights[j] * multivariate_normal.pdf(
                    X, mean=self.means[j], cov=self.covs[j]
                )
            resp /= resp.sum(axis=1, keepdims=True)

            # M-step
            Nk = resp.sum(axis=0)
            for j in range(self.k):
                self.means[j] = (resp[:, j : j + 1].T @ X) / Nk[j]
                diff = X - self.means[j]
                self.covs[j] = (diff.T @ (diff * resp[:, j : j + 1])) / Nk[j]
                self.covs[j] += 1e-6 * np.eye(d)
            self.weights = Nk / n

            # Convergence check
            ll = np.sum(
                np.log(
                    sum(
                        self.weights[j]
                        * multivariate_normal.pdf(X, self.means[j], self.covs[j])
                        for j in range(self.k)
                    )
                )
            )
            if abs(ll - prev_ll) < self.tol:
                print(f"GMM converged at iteration {iteration}")
                break
            prev_ll = ll

        return resp


# =============================================================================
# K-NEAREST NEIGHBORS FROM SCRATCH
# =============================================================================
class KNNClassifier:
    """K-Nearest Neighbors classifier using Euclidean distance."""

    def __init__(self, k: int = 5):
        self.k = k

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X_train = X
        self.y_train = y
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        predictions = []
        for x in X:
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            k_indices = np.argsort(distances)[: self.k]
            k_labels = self.y_train[k_indices]
            values, counts = np.unique(k_labels, return_counts=True)
            predictions.append(values[np.argmax(counts)])
        return np.array(predictions)


# =============================================================================
# DEMO
# =============================================================================
if __name__ == "__main__":
    from sklearn.datasets import load_iris, make_blobs

    # --- PCA Demo ---
    print("=" * 60)
    print("PCA FROM SCRATCH")
    print("=" * 60)
    X_iris, y_iris = load_iris(return_X_y=True)
    pca = PCAFromScratch(n_components=2)
    X_proj = pca.fit_transform(X_iris)
    evr = pca.explained_variance_ratio()
    print(f"Explained variance ratios: {evr}")
    print(f"Total variance captured: {evr.sum():.4f}")

    # --- GMM Demo ---
    print("\n" + "=" * 60)
    print("GMM WITH EM ALGORITHM")
    print("=" * 60)
    X_blobs, _ = make_blobs(n_samples=300, centers=3, random_state=42)
    gmm = GMMFromScratch(k=3)
    responsibilities = gmm.fit(X_blobs)
    print(f"Cluster weights: {gmm.weights.round(3)}")

    # --- KNN Demo ---
    print("\n" + "=" * 60)
    print("KNN FROM SCRATCH")
    print("=" * 60)
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X_iris, y_iris, test_size=0.2, random_state=42
    )
    knn = KNNClassifier(k=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    print(f"KNN Test Accuracy: {accuracy:.4f}")
