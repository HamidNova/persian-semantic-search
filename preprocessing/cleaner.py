from hazm import Normalizer, word_tokenize
import re


normalizer = Normalizer()

STOP_WORDS = [
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
    "های"
]


def clean_text(text):

    text = normalizer.normalize(text)

    text = re.sub(
        r"[^\u0600-\u06FF\s]",
        "",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    words = word_tokenize(text)

    words = [
        word
        for word in words
        if word not in STOP_WORDS
    ]


    return " ".join(words)


def preprocess_dataset(texts):
    cleaned_texts = []

    for text in texts:
        cleaned_texts.append(
            clean_text(text)
        )

    return cleaned_texts
