from sklearn.metrics.pairwise import cosine_similarity


def compare_similarity(
        tfidf_matrix,
        embedding_matrix,
        texts,
        index=0,
        top_k=3
):

    tfidf_scores = cosine_similarity(
        tfidf_matrix
    )[index]

    fasttext_scores = cosine_similarity(
        embedding_matrix
    )[index]

    tfidf_results = []

    fasttext_results = []

    tfidf_indexes = tfidf_scores.argsort()[::-1]

    fasttext_indexes = fasttext_scores.argsort()[::-1]


    for i in tfidf_indexes:
        if i != index:
            tfidf_results.append(
                (
                    texts[i],
                    tfidf_scores[i]
                )
            )

        if len(tfidf_results) == top_k:
            break


    for i in fasttext_indexes:
        if i != index:
            fasttext_results.append(
                (
                    texts[i],
                    fasttext_scores[i]
                )
            )

        if len(fasttext_results) == top_k:
            break


    return (
        tfidf_results,
        fasttext_results
    )


def save_report(
        tfidf_results,
        fasttext_results,
        path
):


    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "===== TF-IDF Results =====\n\n"
        )

        for text, score in tfidf_results:
            f.write(
                f"{text} --> {score}\n"
            )


        f.write(
            "\n\n===== FastText Results =====\n\n"
        )

        for text, score in fasttext_results:
            f.write(
                f"{text} --> {score}\n"
            )
