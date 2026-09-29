import os, io, sys, re, json, csv, time, requests, importlib, tiktoken, chromadb, warnings
# import pandas as pd
# from datetime import datetime
# import numpy as np
# from textwrap import dedent
# from concurrent.futures import ThreadPoolExecutor, as_completed
from sentence_transformers import SentenceTransformer
# from utils.RAG_version import MODEL_NAME


################################################################
# STEP 2. INDEXING (Build or load ChromaDB index) - using SapBERT
################################################################

def flatten_tree(node, acc):
    doc_id = node.get("code") or node.get("block_id")
    if doc_id:
        parts = [f"title: {node.get('title','')}"]
        if node.get("definition"):
            parts.append("definition: " + node["definition"])
        if node.get("parent_id"):
            parts.append(f"parent: {node['parent_id']}")
        acc.append((doc_id, "; ".join(parts)))
    for c in node.get("children", []):
        flatten_tree(c, acc)


def load_or_build_index(model_obj, chroma_db_path, json_file_path, collection_name):
    # Dùng chroma_db_path để khởi tạo Client (Đây là FOLDER)
    client_chroma = chromadb.PersistentClient(path=chroma_db_path)

    try:
        collection = client_chroma.get_collection(collection_name)
        print(f"Loaded existing collection: {collection_name}")
        return collection
    except Exception:
        print(f"Collection {collection_name} not found. Building new one...")

    # Dùng json_file_path để đọc dữ liệu (Đây là FILE)
    if not os.path.exists(json_file_path):
        raise FileNotFoundError(f"JSON file not found: {json_file_path}")

    with open(json_file_path, "r", encoding="utf-8") as f:
        tree = json.load(f)

    docs = []
    for root in tree:
        flatten_tree(root, docs)

    corpus_ids = [doc_id for doc_id, text in docs]
    corpus_texts = [text for doc_id, text in docs]
    print(f"Total documents to index: {len(corpus_texts)}")

    # Tạo Embeddings
    embeddings = model_obj.encode(corpus_texts, batch_size=16, convert_to_numpy=True, show_progress_bar=True)

    # Tạo collection
    try:
        client_chroma.delete_collection(collection_name)
    except:
        pass

    collection = client_chroma.create_collection(collection_name)

    # Tối ưu hóa việc add vào collection (add theo batch nếu dữ liệu lớn)
    # Giới hạn an toàn của ChromaDB thường là 5000 (dưới mức tối đa 5461)
    MAX_BATCH_SIZE = 5000
    total_docs = len(corpus_ids)

    print(f"Adding {total_docs} documents to ChromaDB in batches...")

    for i in range(0, total_docs, MAX_BATCH_SIZE):
        batch_end = min(i + MAX_BATCH_SIZE, total_docs)

        # Lấy từng đoạn dữ liệu
        batch_ids = corpus_ids[i:batch_end]
        batch_embeddings = embeddings[i:batch_end].tolist()
        batch_documents = corpus_texts[i:batch_end]

        # Thêm đợt hiện tại vào collection
        collection.add(
            ids=batch_ids,
            embeddings=batch_embeddings,
            documents=batch_documents
        )
        print(f"  -> Added batch {i} to {batch_end}")

    print(f"Successfully indexed all {total_docs} documents!")

    print(f"Built and saved new collection with SapBERT: {collection_name}")
    return collection


if __name__ == "__main__":
    TREE_JSON = r"C:\Users\VU\Documents\NLP\datasets\ICD-11-MMS.tree.json"
    CHROMA_PATH = r"C:\Users\VU\Documents\NLP\datasets\chroma_icd11"
    COLLECTION_NAME = "icd111"
    MODEL_NAME = SentenceTransformer("cambridgeltl/SapBERT-from-PubMedBERT-fulltext", device="cuda")
    load_or_build_index(MODEL_NAME, CHROMA_PATH, TREE_JSON, COLLECTION_NAME)
