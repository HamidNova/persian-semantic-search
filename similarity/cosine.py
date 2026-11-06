import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def calculate_similarity(matrix):
    """
    Calculate the pairwise cosine similarity matrix.

    Args:
        matrix: A feature or embedding matrix.

    Returns:
        A square cosine similarity matrix.
    """
    return cosine_similarity(matrix)


def get_similar_sentences(
    similarity_matrix,
    index,
    texts,
    top_k=3,
):
    """
    Return the most similar sentences to a selected sentence.

    The sentence at the given index is excluded from the results.

    Args:
        similarity_matrix: Pairwise similarity matrix.
        index: Index of the reference sentence.
        texts: Collection of original sentences.
        top_k: Maximum number of similar sentences to return.

    Returns:
        A list of tuples containing (text, similarity_score).

    Raises:
        ValueError: If the inputs are inconsistent.
        IndexError: If the index is outside the valid range.
    """
    similarity_matrix = np.asarray(similarity_matrix)

    if similarity_matrix.ndim != 2:
        raise ValueError("similarity_matrix must be a 2D matrix.")

    if similarity_matrix.shape[0] != similarity_matrix.shape[1]:
        raise ValueError("similarity_matrix must be square.")

    if len(texts) != similarity_matrix.shape[0]:
        raise ValueError(
            "The number of texts must match the size of the similarity matrix."
        )

    if not 0 <= index < len(texts):
        raise IndexError("index is outside the valid range.")

    if top_k <= 0:
        return []

    scores = similarity_matrix[index]

    similar_indexes = np.argsort(scores)[::-1]

    results = []

    for similar_index in similar_indexes:
        if similar_index == index:
            continue

        results.append(
            (
                texts[similar_index],
                float(scores[similar_index]),
            )
        )

        if len(results) >= top_k:
            break

    return results
