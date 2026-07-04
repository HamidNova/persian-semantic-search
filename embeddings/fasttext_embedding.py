from gensim.models import FastText
import numpy as np


def train_fasttext(texts, vector_size=100):

    sentences = [
        text.split()
        for text in texts
    ]

    model = FastText(
        sentences=sentences,
        vector_size=vector_size,
        window=5,
        min_count=1,
        workers=4,
        seed=42
    )

    return model


def sentence_embedding(sentence, model):

    words = sentence.split()

    vectors = []

    for word in words:
        vectors.append(
            model.wv[word]
        )

    if len(vectors) == 0:
        return np.zeros(
            model.vector_size
        )

    return np.mean(
        vectors,
        axis=0
    )


def create_embedding_matrix(texts, model):

    matrix = []

    for text in texts:
        vector = sentence_embedding(
            text,
            model
        )
        matrix.append(vector)

    return np.array(matrix)


def save_fasttext_model(model, path):
    model.save(path)


def load_fasttext_model(path):
    return FastText.load(path)
