from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def reduce_to_2d(embedding_matrix):
    """
    Reduce an embedding matrix to two dimensions using PCA.

    Args:
        embedding_matrix: A 2D matrix of shape
            (n_samples, n_features).

    Returns:
        A 2D NumPy array with shape (n_samples, 2).

    Raises:
        ValueError: If the input cannot be reduced to two dimensions.
    """
    embedding_matrix = np.asarray(embedding_matrix)

    if embedding_matrix.ndim != 2:
        raise ValueError("embedding_matrix must be a 2D matrix.")

    n_samples, n_features = embedding_matrix.shape

    if n_samples < 2:
        raise ValueError(
            "At least 2 samples are required for 2D PCA."
        )

    if n_features < 2:
        raise ValueError(
            "At least 2 features are required for 2D PCA."
        )

    pca = PCA(n_components=2)

    return pca.fit_transform(embedding_matrix)


def plot_embeddings(
    reduced_embeddings,
    texts,
    save_path=None,
):
    """
    Plot two-dimensional text embeddings.

    Args:
        reduced_embeddings: A 2D embedding matrix with exactly
            two columns.
        texts: Text samples corresponding to the embeddings.
        save_path: Optional path for saving the generated figure.

    Returns:
        The generated Matplotlib figure.

    Raises:
        ValueError: If the inputs have incompatible dimensions.
    """
    reduced_embeddings = np.asarray(reduced_embeddings)

    if reduced_embeddings.ndim != 2:
        raise ValueError(
            "reduced_embeddings must be a 2D matrix."
        )

    if reduced_embeddings.shape[1] != 2:
        raise ValueError(
            "reduced_embeddings must contain exactly 2 columns."
        )

    if len(texts) != len(reduced_embeddings):
        raise ValueError(
            "The number of texts must match the number of embeddings."
        )

    fig, ax = plt.subplots(figsize=(10, 8))

    x = reduced_embeddings[:, 0]
    y = reduced_embeddings[:, 1]

    ax.scatter(x, y)

    for index, (x_value, y_value) in enumerate(zip(x, y), start=1):
        ax.annotate(
            str(index),
            (x_value, y_value),
        )

    ax.set_title("Persian Text Embeddings Visualization")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.grid(True)

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        fig.savefig(
            save_path,
            dpi=150,
            bbox_inches="tight",
        )

    plt.show()

    return fig
