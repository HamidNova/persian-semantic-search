import os
import numpy as np
import pandas as pd

from utils.data_loader import load_text_data
from preprocessing.cleaner import preprocess_dataset
from embeddings.tfidf_embedding import create_tfidf_matrix
from dimensionality.svd_reduction import apply_svd
from embeddings.fasttext_embedding import (
    train_fasttext,
    create_embedding_matrix,
    save_fasttext_model
)
from similarity.cosine import (
    calculate_similarity,
    get_similar_sentences
)
from evaluation.compare_methods import (
    compare_similarity,
    save_report
)
from visualization.plot_embeddings import (
    reduce_to_2d,
    plot_embeddings
)


DATA_PATH = "data/texts.csv"


MODEL_PATH = "models/fasttext_model.bin"
EMBEDDING_PATH = "outputs/embeddings.npy"
SIMILARITY_PATH = "outputs/similarity_results.csv"
REPORT_PATH = "outputs/comparison_report.txt"


def main():

    os.makedirs(
        "models",
        exist_ok=True
    )
    os.makedirs(
        "outputs",
        exist_ok=True
    )


    texts = load_text_data(
        DATA_PATH
    )

    print("="*40)
    print(
        f"Dataset size: {len(texts)}"
    )
    print("="*40)


    cleaned_texts = preprocess_dataset(
        texts
    )


    tfidf_matrix, vectorizer = create_tfidf_matrix(
        cleaned_texts
    )


    print("\nOriginal TF-IDF shape:")
    print(
        tfidf_matrix.shape
    )


    reduced_matrix, svd = apply_svd(
        tfidf_matrix,
        n_components=5
    )


    print("\nAfter SVD:")
    print(
        reduced_matrix.shape
    )


    print("\nExplained variance:")
    print(
        svd.explained_variance_ratio_.sum()
    )


    fasttext_model = train_fasttext(
        cleaned_texts
    )

    save_fasttext_model(
        fasttext_model,
        MODEL_PATH
    )


    embedding_matrix = create_embedding_matrix(
        cleaned_texts,
        fasttext_model
    )


    np.save(
        EMBEDDING_PATH,
        embedding_matrix
    )


    print("\nFastText embedding shape:")
    print(
        embedding_matrix.shape
    )


    tfidf_results, fasttext_results = compare_similarity(
        reduced_matrix,
        embedding_matrix,
        texts,
        index=0
    )


    print("\n===== TF-IDF Results =====")
    for text, score in tfidf_results:
        print(
            text,
            "--->",
            score
        )


    print("\n===== FastText Results =====")
    for text, score in fasttext_results:
        print(
            text,
            "--->",
            score
        )


    save_report(
        tfidf_results,
        fasttext_results,
        REPORT_PATH
    )


    similarity_matrix = calculate_similarity(
        embedding_matrix
    )


    print("\nSimilarity matrix shape:")
    print(
        similarity_matrix.shape
    )


    rows = []

    for i, score in enumerate(similarity_matrix[0]):
        rows.append(
            {
                "sentence": texts[i],
                "similarity": score
            }
        )


    df = pd.DataFrame(rows)


    df.to_csv(
        SIMILARITY_PATH,
        index=False,
        encoding="utf-8-sig"
    )


    results = get_similar_sentences(
        similarity_matrix,
        0,
        texts,
        top_k=3
    )


    print("\nSimilar sentences to:")
    print(
        texts[0]
    )


    for sentence, score in results:
        print(
            "\n",
            sentence,
            "--->",
            score
        )


    reduced_embeddings = reduce_to_2d(
        embedding_matrix
    )


    print("\nVisualization shape:")
    print(
        reduced_embeddings.shape
    )


    plot_embeddings(
        reduced_embeddings,
        texts
    )


    print("\nAll files generated successfully!")


if __name__ == "__main__":

    main()
