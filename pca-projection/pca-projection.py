import numpy as np

def pca_projection(X: list, k: int) -> list:
    """
    Returns the centered data projected onto the top components.
    """
    # Write code here
    X_np = np.asarray(X, dtype=float)
    sample_size = len(X_np)
    feature_ordered_arrays = X_np.T
    centers = np.subtract(X_np, np.asarray([np.mean(row) for row in feature_ordered_arrays],dtype=float))

    cov_matrix = (centers.T @ centers)*(1/(sample_size-1))
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    # sorted_eigenvalvecs = sorted(zip(eigenvalues, eigenvectors), key=lambda x: x[0], reverse=True)
    # eigenvalues_sorted, eigenvectors_sorted = zip(*sorted_eigenvalvecs)
    # eigenvecs = np.asarray(eigenvectors_sorted, dtype=float)
    
    # Reverse them so largest comes first
    eigenvectors = eigenvectors[:, ::-1]
    # Take the first k eigenvectors
    top_eigenvectors = eigenvectors[:, :k]
    # Project data
    projected = centers @ top_eigenvectors
    return projected
    
    