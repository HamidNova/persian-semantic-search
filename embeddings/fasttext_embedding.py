from pathlib import Path

import numpy as np
from gensim.models import FastText


def train_fasttext(texts, vector_size=100):
    """
    Train a FastText model on a collection of preprocessed texts.

    Args:
        texts: A collection of text documents.
        vector_size: Dimensionality of the word vectors.

    Returns:
        A trained Gensim FastText model.

    Raises:
        ValueError: If no valid text is provided.
    """
    if vector_size <= 0:
        raise ValueError("vector_size must be a positive integer.")

    sentences = [
        text.split()
        for text in texts
        if isinstance(text, str) and text.strip()
    ]

    if not sentences:
        raise ValueError(
            "At least one non-empty text is required to train FastText."
        )

    model = FastText(
        sentences=sentences,
        vector_size=vector_size,
        window=5,
        min_count=1,
        workers=4,
        seed=42,
    )

    return model


def sentence_embedding(sentence, model):
    """
    Create a sentence embedding by averaging its word vectors.

    Args:
        sentence: Input text.
        model: A trained FastText model.

    Returns:
        A NumPy array representing the sentence embedding.
    """
    if not isinstance(sentence, str) or not sentence.strip():
        return np.zeros(model.vector_size, dtype=np.float32)

    words = sentence.split()

    vectors = [
        model.wv[word]
        for word in words
    ]

    if not vectors:
        return np.zeros(model.vector_size, dtype=np.float32)

    return np.mean(
        vectors,
        axis=0,
    )


def create_embedding_matrix(texts, model):
    """
    Create sentence embeddings for a collection of texts.

    Args:
        texts: A collection of text documents.
        model: A trained FastText model.

    Returns:
        A NumPy array containing one embedding per input text.
    """
    embeddings = [
        sentence_embedding(text, model)
        for text in texts
    ]

    if not embeddings:
        return np.empty(
            (0, model.vector_size),
            dtype=np.float32,
        )

    return np.asarray(
        embeddings,
        dtype=np.float32,
    )


def save_fasttext_model(model, path):
    """
    Save a trained FastText model.

    Args:
        model: The FastText model to save.
        path: Destination file path.
    """
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save(str(path))


def load_fasttext_model(path):
    """
    Load a saved FastText model.

    Args:
        path: Path to the saved FastText model.

    Returns:
        A loaded FastText model.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"FastText model not found: {path}"
        )

    return FastText.load(str(path))
