import re
from typing import Iterable, List

from hazm import Normalizer, word_tokenize


normalizer = Normalizer()

STOP_WORDS = {
    "است",
    "هست",
    "این",
    "آن",
    "از",
    "به",
    "در",
    "با",
    "برای",
    "و",
    "یک",
    "را",
    "های",
}


def clean_text(text: str) -> str:
    """
    Normalize and clean Persian text for semantic search.

    Steps:
        1. Normalize Persian text using Hazm.
        2. Remove non-Persian characters.
        3. Normalize whitespace.
        4. Tokenize the text.
        5. Remove predefined stopwords.

    Args:
        text: Input text.

    Returns:
        Cleaned text as a space-separated string.
    """
    if not isinstance(text, str):
        return ""

    text = normalizer.normalize(text)

    text = re.sub(
        r"[^\u0600-\u06FF\s]",
        " ",
        text,
    )

    text = re.sub(r"\s+", " ", text).strip()

    words = word_tokenize(text)

    words = [
        word
        for word in words
        if word not in STOP_WORDS
    ]

    return " ".join(words)


def preprocess_dataset(texts: Iterable[str]) -> List[str]:
    """
    Clean a collection of texts.

    Args:
        texts: Iterable containing text samples.

    Returns:
        A list of cleaned texts.
    """
    return [clean_text(text) for text in texts]
