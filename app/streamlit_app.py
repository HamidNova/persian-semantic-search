import sys
from pathlib import Path

import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from embeddings.fasttext_embedding import (
    create_embedding_matrix,
    sentence_embedding,
    train_fasttext,
)
from preprocessing.cleaner import clean_text, preprocess_dataset
from utils.data_loader import load_text_data


DATA_PATH = PROJECT_ROOT / "data" / "texts.csv"


st.set_page_config(
    page_title="Persian Semantic Search",
    page_icon="🔎",
    layout="wide",
)


st.markdown(
    """
    <style>
        .block-container {
            max-width: 1000px;
            padding-top: 3rem;
            padding-bottom: 3rem;
        }

        h1, h2, h3, p, label {
            direction: rtl;
            text-align: right;
        }

        div[data-testid="stTextInput"] input {
            direction: rtl;
            text-align: right;
        }

        div[data-testid="stTextInput"] input::placeholder {
            direction: rtl;
            text-align: right;
        }

        .hero {
            text-align: center;
            direction: rtl;
            padding: 2rem 1rem 2.5rem 1rem;
            margin-bottom: 1.5rem;
        }

        .hero h1 {
            text-align: center;
            margin-bottom: 0.5rem;
        }

        .hero p {
            text-align: center;
            font-size: 1.1rem;
            opacity: 0.8;
        }

        .result-card {
            direction: rtl;
            text-align: right;
            border: 1px solid rgba(128, 128, 128, 0.3);
            border-radius: 12px;
            padding: 1rem 1.25rem;
            margin-bottom: 0.75rem;
        }

        .result-title {
            font-size: 1.05rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }

        .result-score {
            font-size: 0.9rem;
            opacity: 0.75;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_dataset():
    texts = load_text_data(DATA_PATH)
    cleaned_texts = preprocess_dataset(texts)

    return texts, cleaned_texts


@st.cache_resource
def load_model(cleaned_texts):
    return train_fasttext(cleaned_texts)


@st.cache_data
def create_vectors(cleaned_texts):
    model = load_model(cleaned_texts)

    return create_embedding_matrix(
        cleaned_texts,
        model,
    )


def search_similar_texts(
    query,
    model,
    embedding_matrix,
    texts,
    top_k,
):
    cleaned_query = clean_text(query)

    if not cleaned_query:
        return []

    query_vector = sentence_embedding(
        cleaned_query,
        model,
    )

    scores = cosine_similarity(
        query_vector.reshape(1, -1),
        embedding_matrix,
    )[0]

    indexes = np.argsort(scores)[::-1][:top_k]

    results = []

    for index in indexes:
        results.append(
            (
                texts[index],
                float(scores[index]),
            )
        )

    return results


texts, cleaned_texts = load_dataset()

model = load_model(cleaned_texts)

embedding_matrix = create_vectors(cleaned_texts)


st.markdown(
    """
    <div class="hero">
        <h1>🔎 جستجوی معنایی متن فارسی</h1>
        <p>
            یافتن متن‌های مشابه با استفاده از FastText و Cosine Similarity
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


st.divider()


st.markdown(
    "### ✍️ متن مورد نظر خود را وارد کنید"
)

st.caption(
    "سیستم متن شما را تحلیل کرده و مشابه‌ترین متن‌های موجود در مجموعه داده را نمایش می‌دهد."
)


with st.form("search_form"):
    query = st.text_input(
        "متن جستجو",
        placeholder="مثلاً: کاربرد هوش مصنوعی در پزشکی",
        label_visibility="collapsed",
    )

    col1, col2 = st.columns(
        [1, 3]
    )

    with col1:
        top_k = st.selectbox(
            "تعداد نتایج",
            options=[1, 2, 3, 5, 10],
            index=2,
        )

    with col2:
        search_button = st.form_submit_button(
            "🔎 جستجو",
            use_container_width=True,
        )


st.divider()


st.markdown(
    f"💡 **مجموعه داده فعلی شامل {len(texts)} متن فارسی است.**"
)


if search_button:
    if not query.strip():
        st.warning(
            "لطفاً ابتدا یک متن برای جستجو وارد کنید."
        )

    else:
        results = search_similar_texts(
            query=query,
            model=model,
            embedding_matrix=embedding_matrix,
            texts=texts,
            top_k=top_k,
        )

        if results:
            st.markdown(
                "## 🎯 نتایج مشابه"
            )

            st.caption(
                f"{len(results)} نتیجه بر اساس میزان شباهت معنایی پیدا شد."
            )

            for rank, (text, score) in enumerate(
                results,
                start=1,
            ):
                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="result-title">
                            {rank}. {text}
                        </div>
                        <div class="result-score">
                            میزان شباهت: {score:.4f}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        else:
            st.info(
                "برای این متن نتیجه‌ای پیدا نشد."
            )
