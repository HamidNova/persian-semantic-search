from sklearn.decomposition import TruncatedSVD

def apply_svd(matrix, n_components=5):

    svd = TruncatedSVD(
        n_components=n_components,
        random_state=42
    )

    reduced_matrix = svd.fit_transform(matrix)

    return reduced_matrix, svd
