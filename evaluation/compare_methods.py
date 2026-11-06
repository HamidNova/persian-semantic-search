from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def _get_top_results(scores, texts, index, top_k):
    """Return the top similar texts excluding the reference text."""
    sorted_indexes = np.argsort(scores)[::-1]

    results = []

    for similar_index in sorted_indexes:
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


def compare_similarity(
    tfidf_matrix,
    embedding_matrix,
    texts,
    index=0,
    top_k=3,
):
    """
    Compare TF-IDF and FastText similarity results.

    Args:
        tfidf_matrix: TF-IDF feature matrix.
        embedding_matrix: Sentence embedding matrix.
        texts: Collection of original texts.
        index: Index of the reference text.
        top_k: Maximum number of similar texts to return.

    Returns:
        A tuple containing TF-IDF and FastText similarity results.
    """
    embedding_matrix = np.asarray(embedding_matrix)

    if len(tfidf_matrix.shape) != 2:
        raise ValueError("tfidf_matrix must be a 2D matrix.")

    if embedding_matrix.ndim != 2:
        raise ValueError("embedding_matrix must be a 2D matrix.")

    n_texts = len(texts)

    if tfidf_matrix.shape[0] != n_texts:
        raise ValueError(
            "The number of TF-IDF rows must match the number of texts."
        )

    if embedding_matrix.shape[0] != n_texts:
        raise ValueError(
            "The number of embedding rows must match the number of texts."
        )

    if not 0 <= index < n_texts:
        raise IndexError("index is outside the valid range.")

    if top_k <= 0:
        return [], []

    tfidf_scores = cosine_similarity(
        tfidf_matrix
    )[index]

    fasttext_scores = cosine_similarity(
        embedding_matrix
    )[index]

    tfidf_results = _get_top_results(
        tfidf_scores,
        texts,
        index,
        top_k,
    )

    fasttext_results = _get_top_results(
        fasttext_scores,
        texts,
        index,
        top_k,
    )

    return tfidf_results, fasttext_results


def save_report(
    tfidf_results,
    fasttext_results,
    path,
):
    """
    Save TF-IDF and FastText comparison results to a text file.

    Args:
        tfidf_results: Results generated using TF-IDF.
        fasttext_results: Results generated using FastText.
        path: Destination file path.
    """
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            "===== TF-IDF Results =====\n\n"
        )

        for text, score in tfidf_results:
            file.write(
                f"{text} --> {score:.4f}\n"
            )

        file.write(
            "\n===== FastText Results =====\n\n"
        )

        for text, score in fasttext_results:
            file.write(
                f"{text} --> {score:.4f}\n"
            )
