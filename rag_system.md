# RAG检索增强生成系统

## 1. RAG核心概念

**RAG（Retrieval-Augmented Generation，检索增强生成）** 是将外部知识库检索与大语言模型（LLM）生成能力结合的技术架构，通过在生成前检索相关文档片段，显著提升模型的知识覆盖范围、准确性和时效性。

### 为什么需要RAG？

**LLM的局限性**：
- 知识截止日期固定，无法获取最新信息
- 幻觉问题（Hallucination）：模型会编造不存在的事实
- 领域知识不足：预训练数据难以覆盖所有垂直领域
- 上下文长度限制：无法一次性输入整个知识库

**RAG的优势**：
- ✅ 实时知识更新：外部数据库可随时更新
- ✅ 降低幻觉：基于检索到的真实文档生成答案
- ✅ 成本低：无需重新训练大模型
- ✅ 可溯源：生成内容可追溯到具体文档来源

---

## 2. RAG系统架构

```
用户查询 (Query)
    ↓
【1. 查询处理】
    - Query改写/扩展
    - 关键词提取
    - 意图识别
    ↓
【2. 文档检索】
    - 向量检索 (Vector Search)
    - 关键词检索 (BM25)
    - 混合检索 (Hybrid)
    ↓
【3. 重排序 (Rerank)】
    - 相关性打分
    - 多样性筛选
    - Top-K选择
    ↓
【4. 上下文构建】
    - 文档片段拼接
    - Prompt模板填充
    - Token数量控制
    ↓
【5. LLM生成】
    - 基于检索内容生成答案
    - 引用来源标注
    ↓
用户答案 (Answer)
```

---

## 3. 核心技术组件

### 3.1 文档分块（Chunking）

**目标**：将长文档切分为适合检索的最小语义单元。

**常见策略**：
- **固定长度分块**：每512 tokens切分一次（简单但可能破坏语义）
- **语义分块**：按段落/句子边界切分，保留完整语义
- **重叠分块**：相邻chunk保留50-100 tokens重叠，避免信息断裂
- **层次化分块**：文档→章节→段落→句子的多层索引

**代码示例**（语义分块）：
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "！", "？", " ", ""]
)
chunks = splitter.split_text(document)
```

---

### 3.2 向量化（Embedding）

**目标**：将文本转为高维向量（如768维），在向量空间中度量语义相似度。

**常用模型**：
| 模型 | 维度 | 特点 |
|------|------|------|
| text-embedding-ada-002 | 1536 | OpenAI官方，多语言效果好 |
| bge-large-zh-v1.5 | 1024 | 中文专用，FlagEmbedding出品 |
| m3e-base | 768 | 轻量级中文模型 |
| sentence-transformers | 384-1024 | 开源社区标杆 |

**代码示例**：
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('BAAI/bge-large-zh-v1.5')
embeddings = model.encode(chunks)
```

---

### 3.3 向量数据库

**目标**：高效存储和检索高维向量。

**主流方案对比**：
| 数据库 | 索引算法 | 特点 | 适用场景 |
|--------|----------|------|----------|
| **Pinecone** | HNSW | 云原生，开箱即用 | 快速原型开发 |
| **Milvus** | IVF/HNSW | 开源，支持分布式 | 大规模生产环境 |
| **Weaviate** | HNSW | 支持多模态检索 | 图文混合场景 |
| **FAISS** | IVF/PQ | Meta开源，极致性能 | 本地部署 |
| **Chroma** | - | 轻量级，易集成 | 小规模应用 |
| **VikingDB** | - | 火山引擎，中文优化 | 企业级应用 |

**FAISS核心算法**：
- **IVF（倒排索引）**：聚类+索引，牺牲少量精度换速度
- **HNSW（层次图）**：多层图结构，检索精度高但内存占用大
- **PQ（乘积量化）**：向量压缩，降低存储和计算成本

---

### 3.4 检索策略

#### 向量检索（Dense Retrieval）
- 优点：捕捉语义相似性，支持模糊匹配
- 缺点：对关键词精确匹配不敏感

```python
query_vector = model.encode(query)
results = vector_db.search(query_vector, top_k=10)
```

#### 关键词检索（Sparse Retrieval - BM25）
- 优点：精确匹配，支持布尔逻辑
- 缺点：无法理解语义

```python
from rank_bm25 import BM25Okapi

bm25 = BM25Okapi(corpus_tokens)
scores = bm25.get_scores(query_tokens)
```

#### 混合检索（Hybrid Search）
- **倒数排名融合（RRF）**：
  ```python
  score_hybrid = 0.7 * score_vector + 0.3 * score_bm25
  ```
- **Cohere Rerank**：用专门的重排序模型对初排结果重新打分

---

## 4. 高级技术

### 4.1 查询改写（Query Rewriting）

**问题**：用户查询往往简短且模糊，直接检索效果差。

**解决方案**：
- **HyDE（假设性文档嵌入）**：让LLM先生成"假设的完美答案"，用假设答案的向量去检索
  ```python
  hypothetical_doc = llm.generate(f"请详细回答：{query}")
  query_vector = model.encode(hypothetical_doc)
  ```
- **Query扩展**：同义词扩展、问题重构
  ```
  原始："RAG是什么"
  扩展："RAG定义 检索增强生成原理 RAG工作流程"
  ```

---

### 4.2 上下文窗口管理

**问题**：检索到的文档总长度超过LLM上下文限制（如GPT-4为8K tokens）。

**策略**：
- **滑动窗口**：将长文档按窗口切分，每个窗口独立检索
- **重排序Top-K**：只保留最相关的3-5个chunk
- **摘要压缩**：用LLM提前总结长文档，检索时用摘要

---

### 4.3 多跳推理（Multi-Hop Reasoning）

**问题**：复杂问题需要多次检索才能回答。

**示例**：
```
问题："GPT-4的注意力机制与BERT有何不同？"
第一跳：检索"GPT-4架构" → 找到"单向注意力"
第二跳：检索"BERT架构" → 找到"双向注意力"
合成答案：对比两者差异
```

**实现**：ReACT框架 / 图神经网络推理链

---

## 5. 性能优化

### 5.1 检索速度优化

| 技术 | 说明 | 加速比 |
|------|------|--------|
| **向量量化** | Float32→Int8 | 4x内存↓，2x速度↑ |
| **ANN算法** | HNSW/IVF替代暴力搜索 | 10-100x |
| **GPU加速** | FAISS GPU版本 | 5-10x |
| **缓存热点查询** | Redis缓存高频query结果 | ∞（命中时） |

---

### 5.2 检索质量优化

| 方法 | 描述 | 效果提升 |
|------|------|----------|
| **微调Embedding模型** | 在领域数据上微调 | +10-20% Recall@10 |
| **负样本挖掘** | 对比学习训练 | +5-10% MRR |
| **元数据过滤** | 按时间/来源/标签预筛选 | 减少噪声 |
| **迭代检索** | 根据第一轮结果优化query | +15% 准确率 |

---

## 6. 评估指标

### 6.1 检索阶段指标
- **Recall@K**：Top-K结果中包含正确答案的比例
- **MRR（Mean Reciprocal Rank）**：正确答案排名的倒数平均值
- **NDCG**：考虑排序质量的归一化折损累积增益

### 6.2 生成阶段指标
- **准确性**：答案事实正确率（人工标注）
- **相关性**：生成内容与检索文档的一致性
- **幻觉率**：模型编造内容的比例
- **引用率**：答案可追溯到检索文档的比例

---

## 7. 实战案例

### 案例1：企业内部知识库问答

**需求**：员工查询公司政策、技术文档。

**方案**：
```python
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

vectorstore = Chroma(embedding_function=embeddings)
qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(model="gpt-4"),
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5})
)

answer = qa_chain.run("公司年假政策是什么？")
```

---

### 案例2：代码库检索（Code RAG）

**特殊处理**：
- 用AST解析代码结构，按函数/类切分
- 保留代码注释和函数签名
- 支持跨文件依赖检索

**工具**：GitHub Copilot底层技术栈

---

## 8. 常见面试题

### Q1：RAG与Fine-tuning有什么区别？
**答案**：
- **RAG**：外挂知识库，实时检索，成本低，适合知识频繁更新的场景
- **Fine-tuning**：将知识"烧录"到模型参数中，推理快，适合固定领域任务

**对比表**：
| 维度 | RAG | Fine-tuning |
|------|-----|-------------|
| 知识更新 | 实时 | 需重新训练 |
| 成本 | 低 | 高（需GPU训练） |
| 推理速度 | 慢（检索开销） | 快 |
| 幻觉风险 | 低（有检索依据） | 中 |

---

### Q2：如何解决RAG检索不到相关文档的问题？
**答案**：
1. **Query改写**：用HyDE或同义词扩展
2. **降低检索阈值**：增加Top-K的K值
3. **检查Embedding质量**：可能需要微调嵌入模型
4. **文档分块问题**：chunk太大或太小都会影响召回
5. **混合检索**：结合BM25和向量检索

---

### Q3：百万级文档的RAG系统如何设计？
**答案**（参考字节面试题）：
1. **分层索引**：
   - 第一层：文档级别的粗排（按目录/标签）
   - 第二层：chunk级别的精排
2. **分布式向量库**：Milvus集群 + Kafka消息队列
3. **增量更新**：文档更新时只重建变化部分的索引
4. **缓存策略**：Redis缓存高频查询的Top-K结果
5. **异步检索**：用户查询时先返回缓存，后台异步更新

---

### Q4：如何评估RAG系统的效果？
**答案**：
1. **离线评估**：
   - 构建问答对测试集（100-1000条）
   - 计算Recall@10、MRR、NDCG
2. **在线评估**：
   - A/B测试用户点击率
   - 人工标注答案质量（1-5分）
3. **badcase分析**：
   - 检索失败：embedding模型问题
   - 生成偏差：Prompt模板问题
   - 幻觉：需要加强事实校验

---

### Q5：RAG中的Rerank是什么？为什么需要？
**答案**：
- **定义**：对初排的Top-K结果用更精细的模型重新打分排序
- **必要性**：
  - 向量检索是粗排（速度优先），可能有误判
  - Rerank模型（如Cross-Encoder）计算query与document的交互特征，更精准但慢
- **实现**：
  ```python
  from sentence_transformers import CrossEncoder
  reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')
  scores = reranker.predict([(query, doc) for doc in candidates])
  ```

---

## 9. 实战工具推荐

### 框架选择
- **LangChain**：生态完善，适合快速开发
- **LlamaIndex**：专注于索引构建，文档处理能力强
- **Haystack**：可定制性高，适合生产环境

### 向量数据库选型
- **小型项目**：Chroma / FAISS
- **企业级**：Milvus / Pinecone / Weaviate
- **国内云服务**：VikingDB（火山引擎）/ DashVector（阿里云）

---

## 10. 学习路径

### 初级（1-2周）
- [ ] 理解RAG基本流程
- [ ] 用LangChain搭建简单问答系统
- [ ] 掌握FAISS基本使用

### 中级（3-4周）
- [ ] 对比不同Embedding模型效果
- [ ] 实现混合检索（Vector + BM25）
- [ ] 学习Prompt工程优化生成质量

### 高级（5-6周）
- [ ] 微调Embedding模型
- [ ] 设计多跳推理系统
- [ ] 性能优化：缓存、异步、分布式

---

## 参考资料

- [RAG论文原文](https://arxiv.org/abs/2005.11401) - Lewis et al., 2020
- [LangChain官方文档](https://python.langchain.com/)
- [FAISS教程](https://github.com/facebookresearch/faiss/wiki)
- [向量数据库对比](https://github.com/erikbern/ann-benchmarks)
- 腾讯云开发者社区：RAG如何增强LLM的能力
- CSDN博客：2025年向量数据库应用实践
