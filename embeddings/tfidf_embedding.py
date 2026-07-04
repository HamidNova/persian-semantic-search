from sklearn.feature_extraction.text import TfidfVectorizer


def create_tfidf_matrix(texts):

    vectorizer = TfidfVectorizer()

    matrix = vectorizer.fit_transform(texts)

    return matrix, vectorizer
