from collections.abc import Iterable

from sklearn.feature_extraction.text import TfidfVectorizer


def create_tfidf_matrix(texts: Iterable[str]):
    """
    Create a TF-IDF feature matrix from a collection of texts.

    Args:
        texts: A collection of text documents.

    Returns:
        A tuple containing the TF-IDF sparse matrix and the fitted vectorizer.

    Raises:
        ValueError: If no valid non-empty texts are provided.
    """
    texts = [
        str(text).strip()
        for text in texts
        if isinstance(text, str) and text.strip()
    ]

    if not texts:
        raise ValueError(
            "At least one non-empty text is required to create a TF-IDF matrix."
        )

    vectorizer = TfidfVectorizer()

    matrix = vectorizer.fit_transform(texts)

    return matrix, vectorizer
