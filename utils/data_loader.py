import pandas as pd


def load_text_data(path):

    data = pd.read_csv(path)
    texts = data["text"].tolist()
    return texts


def load_dataframe(path):

    return pd.read_csv(path)
