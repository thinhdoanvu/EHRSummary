import os
import json
import re
import time
import torch
import chromadb
from sentence_transformers import models, SentenceTransformer


def clean_and_deduplicate(input_data):
    """
    Xử lý thẻ <think>, loại bỏ Note, và deduplicate cho cả String hoặc List.
    """
    if not input_data:
        return "NA"

    # Nếu là List, chuyển về String để dùng Regex xử lý chung
    if isinstance(input_data, list):
        text = "; ".join(input_data)
    else:
        text = str(input_data)

    if text.upper() == "NA":
        return "NA"

    # 1. Loại bỏ <think>...</think>
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

    # 2. Loại bỏ Note: ... hoặc các giải thích thừa của AI
    text = re.sub(r'(?i)note:.*', '', text).strip()

    # 3. Tách và dọn dẹp từng item
    # Regex tách theo dấu ; hoặc , hoặc xuống dòng
    items = [i.strip() for i in re.split(r'[;,\n]', text) if i.strip()]

    # 4. Deduplicate (giữ nguyên thứ tự)
    seen = set()
    unique_items = []
    for item in items:
        norm = item.upper()
        if norm not in seen and norm != "NA":
            unique_items.append(item)
            seen.add(norm)

    return "; ".join(unique_items) if unique_items else "NA"

class ICD11Council:
    def __init__(self, client_llm, model_manager, tree_path, db_path, qualifier_engine, collection_name="icd11_cls_v2"):
        self.client = client_llm
        self.mm = model_manager
        self.tree_path = tree_path
        self.db_path = db_path
        self.collection_name = collection_name
        self.active_idx = 0

        # 1. Khởi tạo Qualifier Engine (Regex Postcoding)
        self.qualifier_tool = qualifier_engine

        # 2. Khởi tạo Mô hình SapBERT
        print("⏳ Loading SapBERT model...")
        self.model_sapbert = self._init_sapbert()

        # 3. Khởi tạo/Load ChromaDB Collection
        self.chroma_client = chromadb.PersistentClient(path=self.db_path)
        self.collection = self._load_or_build_index()

    def _init_sapbert(self):
        """Khởi tạo pipeline SapBERT tối ưu cho y tế"""
        word_embedding_model = models.Transformer('cambridgeltl/SapBERT-from-PubMedBERT-fulltext')
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='cls')
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        return model.to('cuda') if torch.cuda.is_available() else model

    def _flatten_tree(self, node, acc):
        """Hàm đệ quy làm phẳng cây ICD-11 để chuẩn bị indexing"""
        doc_id = node.get("code") or node.get("block_id")
        if doc_id:
            # Tạo văn bản giàu ngữ nghĩa để embedding chính xác hơn
            parts = [f"title: {node.get('title', '') or node.get('label', '')}"]
            if node.get("definition"):
                parts.append("definition: " + node["definition"])
            if node.get("parent_id"):
                parts.append(f"parent: {node['parent_id']}")

            acc.append((doc_id, "; ".join(parts)))

        # Duyệt qua các nút con (hỗ trợ cả 'children' và 'ChildNodes')
        children = node.get("children") or node.get("ChildNodes") or []
        for c in children:
            self._flatten_tree(c, acc)

    def _load_or_build_index(self):
        """Tự động kiểm tra và xây dựng database nếu chưa tồn tại"""
        try:
            # Thử load collection hiện có
            collection = self.chroma_client.get_collection(self.collection_name)
            if collection.count() > 0:
                print(f"✅ Loaded existing collection: {self.collection_name} ({collection.count()} records)")
                return collection
        except Exception:
            print(f"⚠️ Collection {self.collection_name} not found or empty. Building new one...")

        # Nếu không có, bắt đầu build
        if not os.path.exists(self.tree_path):
            raise FileNotFoundError(f"❌ Không tìm thấy file JSON nguồn tại: {self.tree_path}")

        with open(self.tree_path, "r", encoding="utf-8") as f:
            tree_data = json.load(f)
            # Nếu JSON là một list các root nodes
            if not isinstance(tree_data, list):
                tree_data = [tree_data]

        docs_to_index = []
        for root in tree_data:
            self._flatten_tree(root, docs_to_index)

        print(f"📦 Total nodes found: {len(docs_to_index)}. Creating embeddings...")

        corpus_ids = [d[0] for d in docs_to_index]
        corpus_texts = [d[1] for d in docs_to_index]

        # Tạo Embeddings theo batch để tránh tràn RAM
        embeddings = self.model_sapbert.encode(corpus_texts, batch_size=16, show_progress_bar=True)

        # Xóa và tạo mới collection
        try:
            self.chroma_client.delete_collection(self.collection_name)
        except:
            pass
        collection = self.chroma_client.create_collection(self.collection_name)

        # Thêm vào ChromaDB theo batch (giới hạn an toàn 5000)
        MAX_BATCH = 5000
        for i in range(0, len(corpus_ids), MAX_BATCH):
            end = i + MAX_BATCH
            collection.add(
                ids=corpus_ids[i:end],
                embeddings=embeddings[i:end].tolist(),
                documents=corpus_texts[i:end],
                metadatas=[{"title": t.split(";")[0]} for t in corpus_texts[i:end]]
            )
            print(f"  -> Progress: {min(end, len(corpus_ids))}/{len(corpus_ids)}")

        print(f"🚀 Database built successfully: {self.collection_name}")
        return collection

    def _get_best_stem_code(self, term, candidates):
        """LLM chọn mã Stem Code phù hợp nhất từ Top-K candidates"""
        system_prompt = (
            "You are a Medical Coding Expert. Select the most accurate ICD-11 Stem Code.\n"
            "Rules: Output ONLY the code and label in format: CODE (LABEL). Example: RA01.0 (COVID-19)"
        )
        user_prompt = f"Term: {term}\nCandidates: {json.dumps(candidates, ensure_ascii=False)}"

        while True:
            model_name = self.mm.get_model("CHAIRMAN", attempt=self.active_idx)
            if not model_name: return "N/A"

            try:
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "system", "content": system_prompt},
                              {"role": "user", "content": user_prompt}],
                    **self.mm.get_params("CHAIRMAN"),
                    timeout=30
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                if "resources" in str(e).lower():
                    print(f"❌ {model_name} overloaded, switching...")
                    self.active_idx += 1
                else:
                    return "N/A"

    def get_icd11_code(self, disease_text, summary_text):
        disease_text = clean_and_deduplicate(disease_text)
        if disease_text == "NA":
            return []

        # 1. Postcodes
        found_quals = self.qualifier_tool.extract_from_text(summary_text)
        mapped_postcodes = self.qualifier_tool.map_to_codes(found_quals)
        postcode_list = []
        for cat in mapped_postcodes:
            postcode_list.extend(mapped_postcodes[cat])
        postcode_list = list(dict.fromkeys(postcode_list))

        # 2. Mapping từng bệnh
        diseases = [d.strip() for d in disease_text.split(";") if d.strip()]
        results = []

        for d in diseases:
            query_emb = self.model_sapbert.encode([d], normalize_embeddings=True).tolist()
            search_res = self.collection.query(query_embeddings=query_emb, n_results=5)

            candidates = []
            for i in range(len(search_res["ids"][0])):
                candidates.append({
                    "code": search_res["ids"][0][i],
                    "label": search_res["metadatas"][0][i]["title"]
                })

            best_stem = self._get_best_stem_code(d, candidates)

            if best_stem and best_stem.upper() not in ["N/A", "NA"]:
                stem_only = best_stem.split(" ")[0]
                label_only = best_stem[best_stem.find("("):] if "(" in best_stem else f"({d})"
                cluster = stem_only
                if postcode_list:
                    cluster += " & " + " & ".join(postcode_list)
                results.append(f"{cluster} {label_only}")
            else:
                results.append("NA")

        # --- Trả về list thay vì chuỗi ---
        return results
