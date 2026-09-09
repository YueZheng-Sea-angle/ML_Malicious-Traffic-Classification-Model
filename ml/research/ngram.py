"""载荷字节 n-gram 特征构造（A/B 实验 B 组新增特征路）。

对象域为加密代理流量，无明文 API 可挖，取每条流的载荷字节做 n-gram 统计信号。
流程参照 MALGRA(Electronics 2020) 与 byte n-gram 文献惯例：
    滑窗计数(n=1..max_n) -> 文档频率 min_df 粗筛 -> chi2 对标签选 top-k
    -> TF-IDF 加权 -> 稀疏矩阵。

只允许在训练集上 fit（vocab 与 TF-IDF 参数），transform 复用同一套参数，
与特征方案中「标准化仅取训练集」的防泄漏口径一致。

依赖 sklearn 与 scipy；二者缺失时本模块不可用（A/B 实验本身就要求 LR/RF）。
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import chi2

from ml.features.extractor import Flow


def payload_blob(flow: Flow, max_packets: int = 16, max_bytes: int = 1024) -> bytes:
    """拼接流内前若干包的载荷字节，作为 n-gram 输入文本。"""
    parts = [p.payload for p in flow.packets[:max_packets]]
    return b"".join(parts)[:max_bytes]


def _sliding(blob: bytes, n: int):
    """逐字节滑窗产出长为 n 的 gram（编码为 int，0..256^n-1）。"""
    if len(blob) < n:
        return
    mask = (1 << (8 * n)) - 1
    gram = 0
    for i, byte in enumerate(blob):
        gram = ((gram << 8) | byte) & mask
        if i >= n - 1:
            yield gram


def count_grams(blob: bytes, max_n: int) -> Counter:
    """单个文本(流)在 n=1..max_n 上的 n-gram 频次表。"""
    counts: Counter = Counter()
    for n in range(1, max_n + 1):
        for gram in _sliding(blob, n):
            counts[gram] += 1
    return counts


def gram_to_hex(gram: int, n: int = 4) -> str:
    """把 int 编码的 gram 还原为十六进制文本（用于报告可读性）。"""
    width = max(2, 2 * n)
    return gram.to_bytes(width, "big").hex()


class ByteNgramVectorizer:
    """字节 n-gram 词表选择器 + TF-IDF 变换器。

    fit(blobs, y): 以文档频率(跨流)粗筛 + chi2 标签相关性选出 top-k 词表，
                   并在训练文本上拟合 TF-IDF。
    transform(blobs): 输出 (n_samples, len(vocab_)) 的稀疏 TF-IDF 矩阵。
    """

    def __init__(
        self,
        max_n: int = 4,
        min_df: int = 2,
        top_k: int = 500,
        max_packets: int = 16,
        max_bytes: int = 1024,
    ) -> None:
        self.max_n = max_n
        self.min_df = min_df
        self.top_k = top_k
        self.max_packets = max_packets
        self.max_bytes = max_bytes
        self.vocab_: List[int] = []
        self._tfidf: Optional[TfidfTransformer] = None

    # ------------------------------------------------------------------ #
    def fit(self, blobs: Sequence[bytes], y: Sequence[int]) -> "ByteNgramVectorizer":
        counters = [self._blob_counter(b) for b in blobs]
        df: Dict[int, int] = {}
        for counts in counters:
            for gram in set(counts):
                df[gram] = df.get(gram, 0) + 1
        ndocs = max(len(counters), 1)
        candidates = [g for g, d in df.items() if d >= self.min_df]
        if not candidates:
            raise ValueError("min_df 过滤后无候选 n-gram，请放宽 min_df 或补充文本")

        col_of = {g: i for i, g in enumerate(candidates)}
        counts = self._to_matrix(counters, col_of, len(candidates))
        score, _ = chi2(counts, np.asarray(y, dtype=np.int64))
        # 得分降序 + 列号升序，保证并列时选择完全确定
        order = np.lexsort((np.arange(len(score)), -score))[: self.top_k]
        self.vocab_ = [candidates[i] for i in order]
        self._tfidf = TfidfTransformer(sublinear_tf=False).fit(self._to_matrix(
            counters, {g: i for i, g in enumerate(self.vocab_)}, len(self.vocab_)
        ))
        return self

    def transform(self, blobs: Sequence[bytes]) -> sparse.csr_matrix:
        if not self.vocab_ or self._tfidf is None:
            raise RuntimeError("ByteNgramVectorizer 必须先 fit")
        counters = [self._blob_counter(b) for b in blobs]
        counts = self._to_matrix(
            counters, {g: i for i, g in enumerate(self.vocab_)}, len(self.vocab_)
        )
        return self._tfidf.transform(counts)

    def fit_transform(self, blobs: Sequence[bytes], y: Sequence[int]) -> sparse.csr_matrix:
        return self.fit(blobs, y).transform(blobs)

    # ------------------------------------------------------------------ #
    def _blob_counter(self, blob: bytes) -> Counter:
        blob = bytes(blob)[: self.max_bytes]
        return count_grams(blob, self.max_n)

    @staticmethod
    def _to_matrix(
        counters: Sequence[Counter], col_of: Dict[int, int], ncols: int
    ) -> sparse.csr_matrix:
        rows, cols, data = [], [], []
        for row, counts in enumerate(counters):
            for gram, freq in counts.items():
                col = col_of.get(gram)
                if col is not None:
                    rows.append(row)
                    cols.append(col)
                    data.append(freq)
        return sparse.csr_matrix(
            (data, (rows, cols)), shape=(len(counters), ncols), dtype=np.float64
        )
