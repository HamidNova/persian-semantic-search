# 🔎 Persian Semantic Search

> **A Persian NLP project for comparing TF-IDF, SVD-based latent representations, and FastText embeddings for text similarity and semantic search.**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![GitHub stars](https://img.shields.io/github/stars/HamidNova/persian-semantic-search?style=social)](https://github.com/HamidNova/persian-semantic-search)

An educational Persian Natural Language Processing (NLP) project that explores different approaches to text representation and similarity search.

The project includes Persian text preprocessing using **Hazm**, document representation with **TF-IDF**, dimensionality reduction using **Truncated SVD**, sentence representations generated with **FastText**, cosine similarity search, PCA visualization, and an interactive **Streamlit** application.

---

## ✨ Features

- 🇮🇷 Persian text preprocessing with Hazm
- 📄 TF-IDF document representation
- 📉 Dimensionality reduction using Truncated SVD
- 🧠 FastText-based sentence embeddings
- 🔍 Cosine similarity search
- 📊 PCA visualization of embedding vectors
- ⚖️ Comparison between TF-IDF/SVD and FastText approaches
- 🌐 Interactive Streamlit search interface
- 💾 Automatic generation of models and output files

---

## 🧠 NLP Pipeline

```text
                 Persian Text
                      │
                      ▼
            Preprocessing (Hazm)
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
       TF-IDF                 FastText
          │                       │
          ▼                       ▼
  Truncated SVD          Sentence Embeddings
          │                       │
          └───────────┬───────────┘
                      ▼
              Cosine Similarity
                      │
                      ▼
              Similarity Search
```

---

## 📁 Project Structure

```text
persian-semantic-search/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   └── texts.csv
│
├── dimensionality/
│   └── svd_reduction.py
│
├── embeddings/
│   ├── fasttext_embedding.py
│   └── tfidf_embedding.py
│
├── evaluation/
│   └── compare_methods.py
│
├── models/
│
├── outputs/
│
├── pictures/
│   ├── pca_visualization.png
│   └── streamlit_search.png
│
├── preprocessing/
│   └── cleaner.py
│
├── similarity/
│   └── cosine.py
│
├── utils/
│   └── data_loader.py
│
├── visualization/
│   └── plot_embeddings.py
│
├── .gitignore
├── main.py
├── requirements.txt
└── README.md
```

---

## 🛠️ Technologies

- Python
- NumPy
- Pandas
- Scikit-learn
- Hazm
- Gensim
- Matplotlib
- Streamlit

---

## 🧩 Algorithms and Techniques

### TF-IDF

TF-IDF represents each document based on the importance of its words within the dataset.

It is useful for identifying lexical and keyword-based similarity between texts.

### Truncated SVD

Truncated Singular Value Decomposition (SVD) reduces the dimensionality of the TF-IDF matrix.

This can help capture latent relationships between terms and documents while producing a lower-dimensional representation.

### FastText

FastText learns word representations based on contextual and subword information.

In this project, sentence embeddings are created by averaging the FastText vectors of the words in each sentence.

### Cosine Similarity

Cosine similarity is used to measure the similarity between vector representations of Persian texts.

### PCA

Principal Component Analysis (PCA) reduces high-dimensional embedding vectors to two dimensions for visualization.

---

## ⚖️ TF-IDF vs SVD vs FastText

| Feature | TF-IDF | TF-IDF + SVD | FastText |
|---|:---:|:---:|:---:|
| Keyword-based similarity | ✅ | ✅ | ⚠️ |
| Dense vector representation | ❌ | ✅ | ✅ |
| Dimensionality reduction | ❌ | ✅ | ❌ |
| Captures latent relationships | ❌ | ⚠️ | ⚠️ |
| Word-level semantic information | ❌ | ⚠️ | ⚠️ |
| Suitable for similarity search | ✅ | ✅ | ✅ |

> **Note:** FastText performance depends heavily on the amount and quality of training data. Since this project uses a relatively small educational dataset, the results are primarily intended for experimentation and comparison.

---

## 🔍 Example Search

### Query

```text
هوش مصنوعی در پزشکی
```

The Streamlit application converts the query into an embedding and compares it with the embeddings of the dataset texts using cosine similarity.

The most similar texts are then displayed with their similarity scores.

---

## 📊 Embedding Visualization

FastText generates **100-dimensional sentence embeddings** in this project.

To visualize the embedding space, the vectors are reduced to two dimensions using PCA.

Points that appear closer together may represent texts with more similar vector representations.

---

## 🖼️ Screenshots

| Streamlit Semantic Search | PCA Visualization |
|---|---|
| ![Streamlit Search](pictures/streamlit_search.png) | ![PCA Visualization](pictures/pca_visualization.png) |

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/HamidNova/persian-semantic-search.git
```

Move into the project directory:

```bash
cd persian-semantic-search
```

Create and activate a virtual environment:

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### Run the Complete NLP Pipeline

```bash
python main.py
```

The main pipeline performs the following steps:

1. Loads the Persian text dataset
2. Preprocesses the texts
3. Creates a TF-IDF matrix
4. Applies Truncated SVD
5. Trains a FastText model
6. Creates sentence embeddings
7. Compares TF-IDF/SVD and FastText similarity results
8. Generates a similarity matrix
9. Saves generated results
10. Visualizes the embedding space using PCA

---

## 🌐 Run the Streamlit Application

```bash
streamlit run app/streamlit_app.py
```

Then open the local address displayed by Streamlit in your browser.

The application allows you to:

- Enter a Persian search query
- Select the number of results
- Perform similarity-based search
- View the most similar texts
- Compare similarity scores

---

## 📤 Generated Outputs

Running the main pipeline generates files locally in the `models/` and `outputs/` directories, including:

- Trained FastText model
- Sentence embedding matrix
- Similarity results
- TF-IDF and FastText comparison report

These generated files are excluded from version control through `.gitignore`.

---

## 🔮 Future Improvements

Possible future extensions include:

- Larger and more diverse Persian datasets
- Pre-trained Persian FastText embeddings
- Persian BERT-based embeddings
- Sentence Transformers (SBERT)
- FAISS-based vector search
- Hybrid retrieval methods
- Search result filtering
- REST API integration
- Evaluation with standard Information Retrieval metrics

---

## 🎯 Project Purpose

This project was developed as an educational exploration of different approaches to **Persian text representation and similarity search**.

Its main goal is to provide a practical comparison between:

- Sparse lexical representations using TF-IDF
- Lower-dimensional latent representations using SVD
- Dense embeddings generated with FastText

---

## 👨‍💻 Author

**Hamidreza Mirzaei**

M.Sc. Student in Computer Engineering — Artificial Intelligence

GitHub: https://github.com/HamidNova  
Email: hamidrezamirzaei8363@gmail.com