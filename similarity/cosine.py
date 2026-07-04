from sklearn.metrics.pairwise import cosine_similarity


def calculate_similarity(matrix):

    similarity_matrix = cosine_similarity(
        matrix
    )

    return similarity_matrix


def get_similar_sentences(
        similarity_matrix,
        index,
        texts,
        top_k=3
):

    scores = similarity_matrix[index]

    similar_indexes = scores.argsort(
        )[::-1]

    results = []

    for i in similar_indexes:
        if i != index:
            results.append(
                (
                    texts[i],
                    scores[i]
                )
            )

        if len(results) == top_k:
            break


    return results
