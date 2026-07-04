import streamlit as st
import numpy as np
import sys
import os


sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)


from utils.data_loader import load_text_data
from preprocessing.cleaner import clean_text
from embeddings.fasttext_embedding import (
    train_fasttext,
    sentence_embedding
)
from sklearn.metrics.pairwise import cosine_similarity


st.markdown(
    """
    <style>
    div[data-testid="stMarkdownContainer"] {
        direction: rtl;
        text-align: right;
    }

    div[data-testid="stTextInput"] input {
        direction: rtl;
        text-align: right;
    }

    </style>
    """,
    unsafe_allow_html=True
)

DATA_PATH = "data/texts.csv"

texts = load_text_data(
    DATA_PATH
)

cleaned_texts = [
    clean_text(text)
    for text in texts
]


@st.cache_resource
def load_model():
    model = train_fasttext(
        cleaned_texts
    )
    return model


model = load_model()


@st.cache_data
def create_vectors():

    vectors = []

    for text in cleaned_texts:

        vectors.append(
            sentence_embedding(
                text,
                model
            )
        )

    return np.array(vectors)


embedding_matrix = create_vectors()


st.title(
    "جستجوی معنایی متن فارسی"
)

st.write(
    "پیاده‌سازی جستجوی معنایی با استفاده از FastText و شباهت کسینوسی"
)

query = st.text_input(
    "متن مورد نظر را وارد کنید:"
)

top_k = st.slider(
    "تعداد نتایج:",
    1,
    10,
    3
)

if st.button("جستجو"):

    if query.strip():

        cleaned_query = clean_text(
            query
        )

        query_vector = sentence_embedding(
            cleaned_query,
            model
        )

        scores = cosine_similarity(
            [query_vector],
            embedding_matrix
        )[0]

        indexes = scores.argsort()[::-1]

        st.subheader(
            "نتایج مشابه:"
        )

        count = 0

        for idx in indexes:

            if count == top_k:
                break

            st.markdown(
                f"""
                <div dir="rtl">
                <h4>
                {count + 1}) {texts[idx]}
                </h4>
                <p>
                Similarity: {scores[idx]:.4f}
                </p>
                </div>
                ---
                """,
                unsafe_allow_html=True
            )

            count += 1
