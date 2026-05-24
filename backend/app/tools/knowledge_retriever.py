"""RAG 知识检索模块，从 ChromaDB 检索专业知识并产出 KnowledgeEvidence。

增强特性:
- TTL 检索缓存: 相似查询复用结果，减少 ChromaDB 调用
- 查询扩展: 规则同义扩展 + 可选 LLM 语义变体生成
- 结果去重: 多查询合并后按 method_name + 内容前缀去重
"""

import hashlib
import os
import re
import time
from collections import OrderedDict

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from app.config.setting import settings
from app.schemas.evidence import KnowledgeEvidence
from app.utils.log_util import logger

# ── 数学建模领域同义词典（规则扩展） ──
_SYNONYM_DICT: dict[str, list[str]] = {
    "多指标决策": ["多属性决策", "综合评价", "TOPSIS 熵权法", "层次分析 AHP"],
    "分类": ["聚类", "判别分析", "Fisher判别", "模式识别", "K-means 分类器"],
    "预测": ["回归预测", "时间序列预测", "灰色预测", "拟合外推"],
    "优化": ["最优化", "规划模型", "整数规划", "动态规划", "NP-hard 启发式"],
    "评价": ["评估", "打分模型", "指标体系", "模糊评价"],
    "关联": ["相关性分析", "影响因素", "回归分析", "相关性检验"],
    "插值": ["拟合", "补全", "缺失值填充", "数据增强"],
    "降维": ["主成分分析", "因子分析", "特征选择", "PCA", "t-SNE"],
    "检验": ["假设检验", "显著性检验", "t检验", "卡方检验", "方差分析"],
    "分配": ["匹配", "指派", "任务调度", "匈牙利算法"],
    "路径": ["路线规划", "最短路径", "Dijkstra", "车辆路径问题"],
    "排队": ["队列模型", "服务系统", "M/M/c", "排队论"],
    "蒙特卡洛": ["Monte Carlo", "随机模拟", "仿真", "Bootstrap"],
    "灵敏度": ["敏感性分析", "参数扰动", "鲁棒性分析", "稳定性检验"],
    "TOPSIS": ["优劣解距离法", "理想解法", "逼近理想解"],
}

# ── 停用词表（轻量去重用） ──
_STOP_WORDS = set(
    "的 是 在 和 与 或 对 等 中 为 了 有 不 也 都 从 被 用 将 上 下"
    .split()
)


class KnowledgeRetriever:
    """增强型知识检索器。

    支持:
    - Dense 检索 (ChromaDB) + Rerank (CrossEncoder)
    - TTL 缓存复用
    - 查询扩展 (规则 + 可选 LLM)
    - 多查询结果去重合并
    """

    def __init__(self) -> None:
        self._collection = None
        self._reranker = None
        self._initialized = False
        # 检索缓存: {cache_key: (timestamp, results)}
        self._cache: OrderedDict[str, tuple[float, list[KnowledgeEvidence]]] = (
            OrderedDict()
        )
        self._cache_max_size = 512  # 最多缓存 512 条查询

    # ── 公开 API ──

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        source_type: str | None = None,
        method_name: str | None = None,
    ) -> list[KnowledgeEvidence]:
        """检索与查询相关的知识证据（含缓存）。

        Args:
            query: 检索查询。
            top_k: 返回结果数量，默认使用配置值。
            source_type: 过滤来源类型。
            method_name: 过滤方法名。

        Returns:
            KnowledgeEvidence 列表。
        """
        await self._ensure_initialized()
        if self._collection is None:
            logger.warning("ChromaDB 未初始化，跳过知识检索")
            return []

        top_k = top_k or settings.RAG_TOP_K

        # 缓存查找
        if settings.RAG_CACHE_ENABLED:
            cache_key = self._make_cache_key(query, top_k, source_type, method_name)
            cached = self._cache_get(cache_key)
            if cached is not None:
                logger.debug(f"RAG 缓存命中: {query[:50]}...")
                return cached

        results = await self._do_retrieve(query, top_k, source_type, method_name)

        # 缓存写入
        if settings.RAG_CACHE_ENABLED:
            self._cache_set(cache_key, results)

        return results

    async def retrieve_with_expansion(
        self,
        query: str,
        top_k: int | None = None,
        context: list[str] | None = None,
        llm=None,  # 可选 LLM，用于语义级查询扩展
    ) -> list[KnowledgeEvidence]:
        """带查询扩展的增强检索。

        1. 原始查询检索
        2. 规则同义词扩展 → 并行检索
        3. 可选 LLM 语义扩展 → 并行检索
        4. 合并 + 去重 + 重排序

        Args:
            query: 原始查询文本。
            top_k: 最终返回数量。
            context: 上下文信息（可选，用于 LLM 扩展时提供背景）。
            llm: 可选 LLM 实例，用于语义级查询变体生成。

        Returns:
            去重合并后的 KnowledgeEvidence 列表。
        """
        top_k = top_k or settings.RAG_TOP_K

        # 生成扩展查询
        expanded_queries = self._rule_expand(query)

        if llm and settings.RAG_EXPANSION_ENABLED:
            try:
                llm_variants = await self._llm_expand(query, context, llm)
                expanded_queries.extend(llm_variants)
                logger.info(f"LLM 查询扩展生成 {len(llm_variants)} 个变体")
            except Exception as e:
                logger.warning(f"LLM 查询扩展失败，使用规则扩展: {e}")

        # 去重扩展查询自身
        seen: set[str] = {query}
        unique_queries: list[str] = [query]
        for eq in expanded_queries:
            norm = self._normalize_query(eq)
            if norm and norm not in seen:
                seen.add(norm)
                unique_queries.append(eq)
        logger.info(
            f"查询扩展: 原始 1 个 → {len(unique_queries)} 个"
            f" (规则 {len(expanded_queries)} 个变体)"
        )

        # 并行检索所有查询
        fetch_k = max(top_k, top_k * 2 // len(unique_queries))
        all_results = []
        for q in unique_queries:
            try:
                results = await self.retrieve(q, top_k=fetch_k)
                all_results.extend(results)
            except Exception as e:
                logger.warning(f"扩展查询检索失败 [{q[:40]}...]: {e}")

        # 合并去重
        merged = self._merge_and_dedupe(all_results, top_k)
        logger.info(
            f"扩展检索完成: {len(unique_queries)} 个查询"
            f" → 粗召 {len(all_results)} 条"
            f" → 去重后 {len(merged)} 条"
        )
        return merged

    # ── 内部检索 ──

    async def _do_retrieve(
        self,
        query: str,
        top_k: int,
        source_type: str | None = None,
        method_name: str | None = None,
    ) -> list[KnowledgeEvidence]:
        """执行实际 ChromaDB 检索（不含缓存逻辑）。"""
        where = self._build_where_filter(source_type, method_name)

        try:
            query_embedding = self._embedding_model.encode(query).tolist()
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k * 2,
                where=where if where else None,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            logger.error(f"ChromaDB 查询失败: {e}")
            return []

        if not results or not results["documents"] or not results["documents"][0]:
            return []

        documents = results["documents"][0]
        metadatas = (
            results["metadatas"][0] if results["metadatas"] else [{}] * len(documents)
        )
        distances = (
            results["distances"][0] if results["distances"] else [0.0] * len(documents)
        )

        # Rerank
        if self._reranker and len(documents) > top_k:
            documents, metadatas, distances = self._rerank(
                query, documents, metadatas, distances, top_k
            )

        evidence_list = []
        for doc, meta, dist in zip(
            documents[:top_k], metadatas[:top_k], distances[:top_k]
        ):
            confidence = max(0.0, 1.0 - dist)
            evidence = KnowledgeEvidence(
                content=doc,
                source_type=meta.get("source_type", "textbook"),
                method_name=meta.get("method_name"),
                source_file=meta.get("source_file"),
                source_url=meta.get("source_url"),
                source_title=meta.get("source_title"),
                confidence=min(1.0, confidence),
                metadata=meta,
            )
            evidence_list.append(evidence)

        logger.info(f"知识检索完成: query={query[:60]}..., 找到 {len(evidence_list)} 条")
        return evidence_list

    async def _ensure_initialized(self) -> None:
        """延迟初始化 ChromaDB 和 Reranker。"""
        if self._initialized:
            return

        try:
            import chromadb
            from sentence_transformers import SentenceTransformer

            self._embedding_model = SentenceTransformer(
                settings.RAG_EMBEDDING_MODEL, local_files_only=True
            )

            client = chromadb.PersistentClient(path=settings.RAG_DB_PATH)
            self._collection = client.get_or_create_collection(
                name="knowledge",
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(f"ChromaDB 初始化完成，路径: {settings.RAG_DB_PATH}")
        except Exception as e:
            logger.error(f"ChromaDB 初始化失败: {e}")
            self._collection = None

        try:
            from sentence_transformers import CrossEncoder

            self._reranker = CrossEncoder(
                settings.RAG_RERANKER_MODEL, local_files_only=True
            )
            logger.info(f"Reranker 加载完成: {settings.RAG_RERANKER_MODEL}")
        except Exception as e:
            logger.warning(f"Reranker 加载失败（将跳过 rerank）: {e}")
            self._reranker = None

        self._initialized = True

    # ── 查询扩展 ──

    def _rule_expand(self, query: str) -> list[str]:
        """基于数学建模同义词典生成查询变体。

        对 query 中出现的每个同义关键词，生成替换/补充变体。
        """
        variants: list[str] = []
        for keyword, synonyms in _SYNONYM_DICT.items():
            if keyword in query:
                # 替换关键词为第一个同义词
                variants.append(query.replace(keyword, synonyms[0]))
                # 用其他同义词替换
                for syn in synonyms[1:3]:  # 最多取 2 个变体
                    variants.append(query.replace(keyword, syn))
        return variants[: settings.RAG_EXPANSION_COUNT * 2]

    async def _llm_expand(
        self, query: str, context: list[str] | None, llm
    ) -> list[str]:
        """使用 LLM 生成语义级查询变体。

        Args:
            query: 原始查询。
            context: 上下文信息（如题目背景）。
            llm: LLM 实例 (需有 chat 方法)。
        """
        ctx_text = "\n".join(context[:3]) if context else ""
        prompt = f"""请将以下检索查询改写为 {settings.RAG_EXPANSION_COUNT} 个语义相关但表述不同的查询变体。
这些变体用于检索数学建模方法知识库，请使用不同的术语和角度描述同一主题。

原始查询: {query}
{f'上下文: {ctx_text[:500]}' if ctx_text else ''}

请直接输出 JSON 数组，每项一个字符串，不要其他内容。
示例: ["查询变体1", "查询变体2", "查询变体3\"]"""

        try:
            response = await llm.chat(
                history=[{"role": "user", "content": prompt}],
                tools=None,
                tool_choice=None,
                agent_name="RAG-Expander",
            )
            content = response.choices[0].message.content
            # 提取 JSON 数组
            match = re.search(r"\[.*?\]", content, re.DOTALL)
            if match:
                import json
                variants = json.loads(match.group())
                if isinstance(variants, list):
                    return [v for v in variants if isinstance(v, str) and v != query][
                        : settings.RAG_EXPANSION_COUNT
                    ]
        except Exception as e:
            logger.warning(f"LLM 查询扩展异常: {e}")
        return []

    # ── 结果合并去重 ──

    def _merge_and_dedupe(
        self, results: list[KnowledgeEvidence], top_k: int
    ) -> list[KnowledgeEvidence]:
        """合并多查询结果并按 method_name + 内容前缀去重。

        保留 confidence 最高的副本，按 confidence 降序排列。
        """
        if not results:
            return []

        # 按 (method_name, content_prefix) 分组，取最高 confidence
        deduped: dict[str, KnowledgeEvidence] = {}
        for ev in results:
            fingerprint = self._fingerprint(ev)
            if fingerprint in deduped:
                if ev.confidence > deduped[fingerprint].confidence:
                    deduped[fingerprint] = ev
            else:
                deduped[fingerprint] = ev

        # 按 confidence 降序 + 原始出处的多样性（优先保留不同 method_name）
        sorted_ev = sorted(
            deduped.values(), key=lambda e: e.confidence, reverse=True
        )
        return sorted_ev[:top_k]

    @staticmethod
    def _fingerprint(evidence: KnowledgeEvidence) -> str:
        """为 KnowledgeEvidence 生成去重指纹。

        组合 method_name + 内容前 80 个非停用字符。
        """
        method = evidence.method_name or "unknown"
        # 提取内容前 N 个非停用字符
        content_words = [
            w for w in evidence.content[:200].split()
            if w not in _STOP_WORDS and len(w) > 1
        ]
        prefix = " ".join(content_words[:80])
        # 对内容前缀做轻量哈希以控制 key 长度
        content_hash = hashlib.md5(prefix.encode()).hexdigest()[:12]
        return f"{method}:{content_hash}"

    @staticmethod
    def _normalize_query(query: str) -> str:
        """对查询做轻量归一化用于去重比较。"""
        return re.sub(r"\s+", "", query)[:100]

    # ── 过滤条件构建 ──

    def _build_where_filter(
        self, source_type: str | None, method_name: str | None
    ) -> dict | None:
        """构建 ChromaDB metadata 过滤条件。"""
        conditions = []
        if source_type:
            if "," in source_type:
                types = [t.strip() for t in source_type.split(",")]
                conditions.append({"source_type": {"$in": types}})
            else:
                conditions.append({"source_type": source_type})
        if method_name:
            conditions.append({"method_name": method_name})

        if len(conditions) == 0:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    # ── Rerank ──

    def _rerank(
        self,
        query: str,
        documents: list[str],
        metadatas: list[dict],
        distances: list[float],
        top_k: int,
    ) -> tuple[list[str], list[dict], list[float]]:
        """使用 CrossEncoder 对检索结果重排序。"""
        pairs = [(query, doc) for doc in documents]
        scores = self._reranker.predict(pairs)

        scored = list(zip(documents, metadatas, distances, scores))
        scored.sort(key=lambda x: x[3], reverse=True)

        reranked_docs = [x[0] for x in scored[:top_k]]
        reranked_metas = [x[1] for x in scored[:top_k]]
        reranked_dists = [1.0 - float(x[3]) for x in scored[:top_k]]

        return reranked_docs, reranked_metas, reranked_dists

    # ── 缓存 ──

    def _make_cache_key(
        self,
        query: str,
        top_k: int,
        source_type: str | None,
        method_name: str | None,
    ) -> str:
        """生成缓存键。"""
        raw = f"{query}:{top_k}:{source_type}:{method_name}"
        return hashlib.md5(raw.encode()).hexdigest()

    def _cache_get(self, key: str) -> list[KnowledgeEvidence] | None:
        """从缓存获取结果（TTL 检查）。"""
        if key not in self._cache:
            return None
        timestamp, results = self._cache[key]
        if time.time() - timestamp > settings.RAG_CACHE_TTL:
            del self._cache[key]
            return None
        return results

    def _cache_set(self, key: str, results: list[KnowledgeEvidence]) -> None:
        """写入缓存，超出容量时淘汰最旧条目。"""
        if len(self._cache) >= self._cache_max_size:
            self._cache.popitem(last=False)  # FIFO 淘汰
        self._cache[key] = (time.time(), results)
        # 确保被访问的 key 移到末尾 (OrderedDict move_to_end)
        self._cache.move_to_end(key)


# 全局单例
knowledge_retriever = KnowledgeRetriever()
