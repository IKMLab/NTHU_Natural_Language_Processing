"""The helper functions for the retriever part of the RAG lab 2.
You should install the packages in `RAG_tutorial_2.ipynb` first.
"""

import re
import json
import pickle
import numpy as np
from numpy.linalg import norm
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi


def data_preprocessing(text):
    # Replace newline characters
    text = text.replace("\n", " ")
    # Remove excessive punctuation (e.g., "!!!" -> "!")
    text = re.sub(r"([.,!?])\1+", r"\1", text)

    # Replace multiple spaces with a single space
    text = re.sub(r"\s+", " ", text)

    # Remove URL,HTML tags
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>", "", text)

    # Remove special characters (keep alphanumeric and basic punctuation)
    text = re.sub(r'[^a-zA-Z0-9\s.,!?\'"-]', "", text)

    text = text.strip()
    return text


def load_text_db(file_path):
    """
    Loads a text database from a JSON file.
    """
    with open(file_path, "r") as f:
        text_db = json.load(f)
    return text_db


# Dense
def cos_sim(a, b):
    """
    Computes the cosine similarity of two vectors.
    """
    return (a @ b.T) / (norm(a) * norm(b))


def dense_ranker(query, vector_db, model):
    """
    Ranks documents using cos_sim for a given query.
    """
    # Encode the query into a vector
    query_vector = model.encode(query)

    # Compute cosine similarity scores for each vector in the database
    scores = [
        {
            "id": doc["id"],
            "text": doc.get(
                "text", None
            ),  # Optional: retrieve document content if available
            "score": cos_sim(query_vector, doc["vector"]),
        }
        for doc in vector_db
    ]

    # Sort the documents by score in descending order
    ranked_docs = sorted(scores, key=lambda x: x["score"], reverse=True)

    return ranked_docs


def load_vector_db(file_path):
    """
    Loads a vector database from a JSON file.
    """
    with open(file_path, "r") as f:
        serialized_data = json.load(f)
        vector_db = [
            {"id": doc["id"], "text": doc["text"], "vector": np.array(doc["vector"])}
            for doc in serialized_data
        ]
    return vector_db


# Sparse
def preprocess_texts(texts):
    """
    Tokenizes and preprocesses text documents.Tokenizes and preprocesses text documents.
    """
    return [word_tokenize(text.lower()) for text in texts]


def build_bm25_index(text_db):
    """
    Builds a BM25 index from a list of text documents.
    """
    # Extract texts and preprocess them
    texts = [doc["text"] for doc in text_db]
    tokenized_corpus = preprocess_texts(texts)

    # Build BM25 index
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, tokenized_corpus


def save_bm25_index(bm25, file_path):
    with open(file_path, "wb") as bm25result_file:
        pickle.dump(bm25, bm25result_file)
    # print(f"BM25 tokenized corpus saved to {file_path}.")


def load_bm25_index(file_path):
    with open(file_path, "rb") as bm25result_file:
        bm25 = pickle.load(bm25result_file)
    # print(f"BM25 tokenized corpus loaded from {file_path}.")

    return bm25


def bm25_ranker(query, bm25, text_db):
    """
    Ranks documents using BM25 for a given query.
    """
    # Tokenize query
    tokenized_query = word_tokenize(query.lower())

    # Get BM25 scores
    scores = bm25.get_scores(tokenized_query)

    # Rank documents by score
    ranked_docs = sorted(
        [
            {"id": doc["id"], "text": doc["text"], "score": scores[i]}
            for i, doc in enumerate(text_db)
        ],
        key=lambda x: x["score"],
        reverse=True,
    )

    # Return results
    return ranked_docs


def hybrid_ranker_rrf(dense_ranked_docs, sparse_ranked_docs, k=60, k_bm=60):
    """
    Combines dense and BM25 ranking results using Reciprocal Rank Fusion (RRF).
    """
    # Create dictionaries for quick look-up of ranks
    dense_ranks = {
        doc["id"]: rank for rank, doc in enumerate(dense_ranked_docs, start=1)
    }
    sparse_ranks = {
        doc["id"]: rank for rank, doc in enumerate(sparse_ranked_docs, start=1)
    }

    # Collect all unique document IDs from both ranking results
    all_doc_ids = set(dense_ranks.keys()).union(sparse_ranks.keys())

    # Compute RRF scores
    rrf_scores = {}
    for doc_id in all_doc_ids:
        dense_rank = dense_ranks.get(
            doc_id, len(dense_ranks) + 1
        )  # Use max_len of sorted + 1 for missing docs
        sparse_rank = sparse_ranks.get(doc_id, len(sparse_ranks) + 1)

        # RRF score formula: 1 / (k + rank)
        if k_bm == 60:
            rrf_scores[doc_id] = (1 / (k + dense_rank)) + (1 / (k + sparse_rank))
        else:
            rrf_scores[doc_id] = (1 / (k + dense_rank)) + (1 / (k_bm + sparse_rank))

    # Combine results and sort by RRF score
    hybrid_ranked_docs = []
    for doc_id, rrf_score in sorted(
        rrf_scores.items(), key=lambda x: x[1], reverse=True
    ):
        # Retrieve document content from either dense_ranked_docs or parse_ranked_docs
        doc_content = next(
            (doc["text"] for doc in dense_ranked_docs if doc["id"] == doc_id),
            next(doc["text"] for doc in sparse_ranked_docs if doc["id"] == doc_id),
        )
        hybrid_ranked_docs.append(
            {"id": doc_id, "text": doc_content, "score": rrf_score}
        )

    return hybrid_ranked_docs


def personal_retriever(
    query, text_db_path, vector_db_path, bm25_db_path, emb_model, topk=3, k_bm=60
):
    """
    Using hybrid method to retrieve the top_k docs from database.
    """
    # Load text_db
    text_db = load_text_db(text_db_path)
    # Load vector_db
    vector_db = load_vector_db(vector_db_path)
    # Load the tokenized corpus and rebuild the BM25 index
    bm25 = load_bm25_index(bm25_db_path)

    topk = 3
    dense_ranked_docs = dense_ranker(query=query, vector_db=vector_db, model=emb_model)
    sparse_ranked_docs = bm25_ranker(query=query, bm25=bm25, text_db=text_db)

    # Rank by using hybrid_ranker_rrk
    hybrid_ranked_docs = hybrid_ranker_rrf(
        dense_ranked_docs=dense_ranked_docs,
        sparse_ranked_docs=sparse_ranked_docs,
        k_bm=k_bm,
    )
    return hybrid_ranked_docs[:topk]
