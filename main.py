from pathlib import Path

import numpy as np
import pandas as pd

from utils.data_loader import load_text_data
from preprocessing.cleaner import preprocess_dataset
from embeddings.tfidf_embedding import create_tfidf_matrix
from dimensionality.svd_reduction import apply_svd
from embeddings.fasttext_embedding import (
    train_fasttext,
    create_embedding_matrix,
    save_fasttext_model,
)
from similarity.cosine import (
    calculate_similarity,
    get_similar_sentences,
)
from evaluation.compare_methods import (
    compare_similarity,
    save_report,
)
from visualization.plot_embeddings import (
    reduce_to_2d,
    plot_embeddings,
)


BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "texts.csv"

MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

MODEL_PATH = MODEL_DIR / "fasttext_model.bin"
EMBEDDING_PATH = OUTPUT_DIR / "embeddings.npy"
SIMILARITY_PATH = OUTPUT_DIR / "similarity_results.csv"
REPORT_PATH = OUTPUT_DIR / "comparison_report.txt"


def main():
    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 40)
    print("Persian Semantic Search")
    print("=" * 40)

    print("\n[1] Loading dataset...")

    texts = load_text_data(
        DATA_PATH
    )

    print(
        f"Dataset size: {len(texts)}"
    )

    print("\n[2] Preprocessing texts...")

    cleaned_texts = preprocess_dataset(
        texts
    )

    print(
        f"Processed texts: {len(cleaned_texts)}"
    )

    print("\n[3] Creating TF-IDF matrix...")

    tfidf_matrix, _ = create_tfidf_matrix(
        cleaned_texts
    )

    print(
        f"TF-IDF matrix shape: "
        f"{tfidf_matrix.shape}"
    )

    print("\n[4] Applying SVD...")

    reduced_matrix, svd = apply_svd(
        tfidf_matrix,
        n_components=5,
    )

    explained_variance = (
        svd.explained_variance_ratio_.sum()
    )

    print(
        f"Reduced matrix shape: "
        f"{reduced_matrix.shape}"
    )

    print(
        f"Explained variance: "
        f"{explained_variance:.4f}"
    )

    print("\n[5] Training FastText model...")

    fasttext_model = train_fasttext(
        cleaned_texts
    )

    save_fasttext_model(
        fasttext_model,
        MODEL_PATH,
    )

    embedding_matrix = create_embedding_matrix(
        cleaned_texts,
        fasttext_model,
    )

    np.save(
        EMBEDDING_PATH,
        embedding_matrix,
    )

    print(
        f"FastText embedding shape: "
        f"{embedding_matrix.shape}"
    )

    print("\n[6] Comparing TF-IDF and FastText...")

    tfidf_results, fasttext_results = (
        compare_similarity(
            tfidf_matrix,
            embedding_matrix,
            texts,
            index=0,
            top_k=3,
        )
    )

    print("\n===== TF-IDF Results =====")

    for text, score in tfidf_results:
        print(
            f"{text} ---> {score:.4f}"
        )

    print("\n===== FastText Results =====")

    for text, score in fasttext_results:
        print(
            f"{text} ---> {score:.4f}"
        )

    save_report(
        tfidf_results,
        fasttext_results,
        REPORT_PATH,
    )

    print(
        f"\nComparison report saved to: "
        f"{REPORT_PATH}"
    )

    print("\n[7] Calculating FastText similarities...")

    similarity_matrix = calculate_similarity(
        embedding_matrix
    )

    print(
        f"Similarity matrix shape: "
        f"{similarity_matrix.shape}"
    )

    rows = []

    for i, score in enumerate(
        similarity_matrix[0]
    ):
        if i == 0:
            continue

        rows.append(
            {
                "sentence": texts[i],
                "similarity": float(score),
            }
        )

    similarity_dataframe = pd.DataFrame(
        rows
    ).sort_values(
        by="similarity",
        ascending=False,
    )

    similarity_dataframe.to_csv(
        SIMILARITY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"Similarity results saved to: "
        f"{SIMILARITY_PATH}"
    )

    results = get_similar_sentences(
        similarity_matrix,
        0,
        texts,
        top_k=3,
    )

    print("\nSimilar sentences to:")
    print(texts[0])

    for sentence, score in results:
        print(
            f"\n{sentence} ---> {score:.4f}"
        )

    print("\n[8] Creating embedding visualization...")

    reduced_embeddings = reduce_to_2d(
        embedding_matrix
    )

    print(
        f"Visualization shape: "
        f"{reduced_embeddings.shape}"
    )

    plot_embeddings(
        reduced_embeddings,
        texts,
    )

    print("\nAll files generated successfully!")


if __name__ == "__main__":
    main()
