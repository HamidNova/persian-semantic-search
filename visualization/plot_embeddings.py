import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def reduce_to_2d(embedding_matrix):

    pca = PCA(
        n_components=2
    )

    reduced = pca.fit_transform(
        embedding_matrix
    )

    return reduced


def plot_embeddings(
        reduced_embeddings,
        texts
):

    plt.figure(
        figsize=(10,8)
    )

    x = reduced_embeddings[:,0]
    y = reduced_embeddings[:,1]


    plt.scatter(
        x,
        y
    )

    for i, text in enumerate(texts):
        plt.annotate(
            str(i+1),
            (x[i], y[i])
        )


    plt.title(
        "Persian Text Embeddings Visualization"
    )

    plt.xlabel(
        "Component 1"
    )

    plt.ylabel(
        "Component 2"
    )

    plt.grid(True)

    plt.show()
