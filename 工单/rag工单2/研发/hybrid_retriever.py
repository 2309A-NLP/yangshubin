# -*- coding: utf-8 -*-
"""
混合检索模块 —— 稠密向量 + 关键词检索 + RRF融合
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
"""

import math
import os
import pickle
from typing import List, Dict
from collections import Counter
from pathlib import Path

from config import TOP_K_RETRIEVAL, TOP_K_RERANK, OUTPUT_DIR
from embedding_model import EmbeddingModel
from vector_store import VectorStore


class _BM25Scorer:
    """轻量BM25，基于词频+逆文档频率计算关键词匹配分数"""

    def __init__(self, docs: List[str]):
        self.docs = docs
        self.N = len(docs)
        self.avgdl = sum(len(d.split()) for d in docs) / max(self.N, 1)
        self.k1, self.b = 1.5, 0.75
        self.df = Counter()
        for d in docs:
            self.df.update(set(d.split()))
        self.idf = {w: math.log((self.N - f + 0.5) / (f + 0.5) + 1) for w, f in self.df.items()}

    def score(self, query: str, doc_idx: int) -> float:
        doc = self.docs[doc_idx]
        doc_len = len(doc.split())
        score = 0.0
        for q in query.split():
            if q not in self.idf:
                continue
            tf = doc.count(q)
            if tf == 0:
                continue
            score += self.idf[q] * (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl))
        return score


class HybridRetriever:
    """
    混合检索器：
    1. 稠密检索（BGE-M3向量 → Milvus）
    2. 关键词检索（BM25，缓存到磁盘）
    3. RRF融合排序
    """

    BM25_CACHE = str(Path(OUTPUT_DIR) / "bm25_cache.pkl")

    def __init__(self):
        self.embedder = EmbeddingModel()
        self.store = VectorStore()
        self._bm25 = None
        self._all_texts = []

    def warm_up(self):
        """预热：提前加载BGE-M3 + 构建/加载BM25缓存"""
        print("[预热] 加载BGE-M3模型...")
        self.embedder._load()
        print("[预热] BGE-M3就绪")
        self._load_bm25_cache()

    def _load_bm25_cache(self):
        """从磁盘加载BM25缓存，不存在则构建"""
        if self._bm25 is not None:
            return

        # 尝试从缓存加载
        if os.path.exists(self.BM25_CACHE):
            print("[BM25] 从缓存加载...")
            with open(self.BM25_CACHE, "rb") as f:
                data = pickle.load(f)
            self._all_texts = data["texts"]
            self._bm25 = _BM25Scorer(self._all_texts)
            self._bm25.df = data["df"]
            self._bm25.idf = data["idf"]
            self._bm25.avgdl = data["avgdl"]
            print(f"[BM25] 缓存加载成功，{len(self._all_texts)}条")
            return

        # 无缓存则从Milvus拉取
        print("[BM25] 构建索引...")
        all_texts = []
        offset = 0
        while True:
            results = self.store.client.query(
                self.store.collection, limit=200, offset=offset, output_fields=["text"]
            )
            if not results:
                break
            all_texts.extend(r["text"] for r in results if "text" in r)
            offset += 200

        self._all_texts = all_texts
        self._bm25 = _BM25Scorer(all_texts)

        # 保存缓存
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(self.BM25_CACHE, "wb") as f:
            pickle.dump({"texts": all_texts, "df": self._bm25.df,
                         "idf": self._bm25.idf, "avgdl": self._bm25.avgdl}, f)
        print(f"[BM25] 构建并缓存完成，{len(all_texts)}条")

    def retrieve(self, question: str, top_k: int = None) -> tuple:
        """混合检索 + RRF融合，返回(融合结果, 稠密结果, 关键词结果)"""
        if top_k is None:
            top_k = TOP_K_RERANK
        self._load_bm25_cache()

        # 1. 稠密检索
        query_vec = self.embedder.encode_query(question)[0].tolist()
        dense_results = self.store.search(query_vec, top_k=TOP_K_RETRIEVAL)
        dense_map = {r["id"]: r for r in dense_results}

        # 2. 关键词检索
        kw_scores = [(i, self._bm25.score(question, i))
                     for i in range(len(self._all_texts))
                     if self._bm25.score(question, i) > 0]
        kw_scores.sort(key=lambda x: x[1], reverse=True)
        kw_results = [{"id": idx, "text": self._all_texts[idx][:300], "score": s}
                      for idx, s in kw_scores[:TOP_K_RETRIEVAL]]

        # 3. RRF融合
        fused = {}
        for rank, r in enumerate(dense_results):
            fused[r["id"]] = fused.get(r["id"], 0) + 1 / (60 + rank + 1)
        for rank, r in enumerate(kw_results):
            fused[r["id"]] = fused.get(r["id"], 0) + 1 / (60 + rank + 1)

        fused_ids = sorted(fused.keys(), key=lambda x: fused[x], reverse=True)[:top_k]
        final = []
        for fid in fused_ids:
            item = dict(dense_map.get(fid, {}))
            item["rrf_score"] = round(fused[fid], 4)
            final.append(item)

        return final, dense_results[:top_k], kw_results[:top_k]


if __name__ == "__main__":
    hr = HybridRetriever()
    hr.warm_up()
    q = "武汉兴图新科电子股份有限公司注册资本是多少？"
    fused, dense, kw = hr.retrieve(q)
    print(f"问题: {q}")
    for i, r in enumerate(fused[:3], 1):
        print(f"  {i}. RRF={r.get('rrf_score',0):.4f} | {r['text'][:80]}...")
