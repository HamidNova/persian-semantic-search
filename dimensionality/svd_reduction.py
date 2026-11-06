import numpy as np
from sklearn.decomposition import TruncatedSVD


def apply_svd(matrix, n_components=5):
    """
    Reduce a feature matrix using Truncated Singular Value Decomposition.

    Args:
        matrix: A 2D feature matrix.
        n_components: Number of dimensions to keep.

    Returns:
        A tuple containing the reduced matrix and the fitted SVD model.

    Raises:
        ValueError: If the input matrix or n_components is invalid.
    """
    if not hasattr(matrix, "shape"):
        matrix = np.asarray(matrix)

    if len(matrix.shape) != 2:
        raise ValueError("matrix must be a 2D matrix.")

    n_samples, n_features = matrix.shape

    if n_samples == 0 or n_features == 0:
        raise ValueError("matrix must not be empty.")

    if not isinstance(n_components, int) or n_components <= 0:
        raise ValueError("n_components must be a positive integer.")

    max_components = min(n_samples, n_features)

    if n_components >= max_components:
        raise ValueError(
            f"n_components must be smaller than {max_components} "
            f"for a matrix with shape {matrix.shape}."
        )

    svd = TruncatedSVD(
        n_components=n_components,
        random_state=42,
    )

    reduced_matrix = svd.fit_transform(matrix)

    return reduced_matrix, svd
