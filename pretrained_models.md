# 预训练模型专题

## 🌟 预训练范式

### 什么是预训练-微调范式？

```
预训练（Pre-training）
    ↓
在大规模无标注数据上训练通用语言表示
    ↓
微调（Fine-tuning）
    ↓
在特定任务的小规模标注数据上调整
    ↓
任务模型
```

**核心优势**：
- 利用大规模无标注数据学习语言知识
- 迁移学习，减少下游任务数据需求
- 提升模型泛化能力

---

## 1️⃣ BERT系列

### BERT（Bidirectional Encoder Representations from Transformers）

**论文**：BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (2018)

**核心特点**：
- 双向编码器（基于Transformer Encoder）
- 适合理解类任务（分类、问答、NER）
- 不适合生成任务

**架构**：
```
Input: [CLS] 我 爱 自 然 语 言 处 理 [SEP]
         ↓
    Embedding Layer (Token + Position + Segment)
         ↓
    12/24 × Transformer Encoder Blocks
         ↓
    Output: Hidden States
```

### BERT预训练任务

#### 1. MLM（Masked Language Model）
- **做法**：随机mask 15%的token，让模型预测被mask的词
- **示例**：
  ```
  原句：我爱自然语言处理
  输入：我[MASK]自然语言处理
  预测：爱
  ```
- **细节**：
  - 80% 替换为 [MASK]
  - 10% 替换为随机词
  - 10% 保持不变
- **目的**：学习上下文表示

#### 2. NSP（Next Sentence Prediction）
- **做法**：预测两个句子是否连续
- **示例**：
  ```
  句A：今天天气真好
  句B：我们去公园玩吧
  标签：IsNext
  
  句A：今天天气真好
  句B：量子力学很难
  标签：NotNext
  ```
- **目的**：学习句子间关系（后续研究表明此任务效果有限）

### BERT变体

| 模型 | 发布时间 | 改进点 |
|------|----------|--------|
| **RoBERTa** | 2019 | 去除NSP、动态masking、更大batch size |
| **ALBERT** | 2019 | 参数共享、减少参数量、句序预测(SOP) |
| **ELECTRA** | 2020 | 判别式预训练、更高效 |
| **DeBERTa** | 2020 | 解耦注意力机制 |

### RoBERTa详解
**改进**：
- 移除NSP任务
- 动态masking（每次epoch mask不同位置）
- 更大的batch size（8K）
- 更多训练数据和更长训练时间
- 使用BPE tokenizer

**结果**：在多数任务上超越BERT

### ALBERT详解
**核心创新**：
- **Factorized Embedding**：词嵌入维度 < 隐层维度
  - BERT: vocab_size × hidden_size
  - ALBERT: vocab_size × embedding_size + embedding_size × hidden_size
- **Cross-layer Parameter Sharing**：所有层共享参数
- **SOP（Sentence Order Prediction）**：替代NSP，判断句子顺序

**优势**：参数量大幅减少，训练速度更快

---

## 2️⃣ GPT系列

### GPT-1（2018）

**核心思想**：生成式预训练 + 任务微调

**架构**：
- 单向Transformer Decoder（只能看左边的上文）
- 12层、768维、12个注意力头

**预训练任务**：语言模型（预测下一个词）
```
输入：我 爱 自 然
预测：语言
```

### GPT-2（2019）

**创新点**：
- 更大规模（1.5B参数）
- Zero-shot learning（无需微调直接做任务）
- 使用task prefix进行提示

**示例**：
```
翻译任务：
输入：translate English to French: Hello
输出：Bonjour

问答任务：
输入：Question: What is AI? Answer:
输出：AI is artificial intelligence...
```

**争议**：OpenAI最初未公开完整模型（担心滥用）

### GPT-3（2020）

**规模**：
- 175B参数（GPT-2的116倍）
- 45TB训练数据

**能力飞跃**：
- **Few-shot Learning**：给几个示例就能做任务
- **In-context Learning**：通过prompt控制行为
- **涌现能力**：算术、代码、推理

**示例**：
```
Few-shot示例：
输入：
海水朝朝朝朝朝朝朝落
→ 读音：海水潮，朝朝潮，朝潮朝落

云长长长长长长长消
→ 读音：
输出：云长，长长长，长长长消
```

### GPT-3.5 / ChatGPT（2022）

**核心技术**：
- **SFT（Supervised Fine-Tuning）**：人工标注对话数据微调
- **RLHF（Reinforcement Learning from Human Feedback）**：
  ```
  1. 收集人类偏好数据（比较不同回复）
  2. 训练奖励模型（Reward Model）
  3. 用PPO算法优化策略
  ```

**效果**：
- 更安全（减少有害输出）
- 更有用（遵循指令）
- 更诚实（承认不知道）

### GPT-4（2023）

**改进**：
- 多模态（图像+文本输入）
- 更长上下文（32K token）
- 更强推理能力
- 通过人类考试（律师资格考试前10%）

---

## 3️⃣ BERT vs GPT 对比

| 维度 | BERT | GPT |
|------|------|-----|
| **架构** | Encoder-only | Decoder-only |
| **注意力** | 双向（看全文） | 单向（只看左边） |
| **预训练** | MLM + NSP | 语言模型（下一词预测） |
| **适合任务** | 理解类（分类、NER、问答） | 生成类（文本生成、翻译） |
| **输入方式** | 完整句子 | 从左到右逐词 |
| **典型应用** | 情感分析、文本分类 | 对话、写作、代码生成 |

**类比**：
- BERT：完形填空（看全文猜空格）
- GPT：续写作文（只看前文写后文）

---

## 4️⃣ T5（Text-to-Text Transfer Transformer）

**核心思想**：所有NLP任务统一为文本生成

**架构**：完整的Encoder-Decoder Transformer

**任务格式化**：
```
翻译：translate English to German: Hello → Hallo
分类：sentiment: This movie is great → positive
问答：question: What is AI? context: ... → AI is...
摘要：summarize: [long text] → [short summary]
```

**优势**：
- 统一框架处理所有任务
- 便于多任务学习
- 性能优异

---

## 5️⃣ 其他重要模型

### XLNet（2019）
- 结合BERT双向性和GPT自回归
- 使用Permutation Language Modeling
- 解决BERT的[MASK]不一致问题

### BART（2019）
- Seq2Seq架构（Encoder-Decoder）
- 预训练：加噪（删除、打乱、mask）+ 去噪
- 适合生成和理解任务

### ERNIE（百度，2019）
- 知识增强的预训练
- Entity Masking（mask整个实体）
- Phrase Masking（mask短语）

---

## 6️⃣ 面试高频问题

### Q1: BERT为什么不能做生成任务？
**答**：
- BERT是双向编码器，训练时看到了全文
- 生成任务需要逐词生成，不能看到未来的词
- BERT的[MASK]机制也不适合生成
- 但可以用BERT做生成的辅助（如条件生成的编码器）

### Q2: GPT为什么是单向的？
**答**：
- 为了保持因果性（Causality）
- 生成时只能看到已生成的部分
- 训练和推理保持一致
- 避免信息泄露

### Q3: BERT的[MASK]会导致什么问题？
**答**：
- **预训练-微调不一致**：预训练有[MASK]，微调没有
- **独立性假设**：被mask的词之间相互独立
- **解决**：XLNet用排列语言模型、ELECTRA用判别式训练

### Q4: RLHF的作用是什么？
**答**：
- 让模型输出更符合人类偏好
- 减少有害、错误输出
- 提升指令跟随能力
- 核心：奖励模型 + PPO优化

### Q5: 为什么GPT-3能做In-context Learning？
**答**：
- 规模效应（175B参数）
- 训练数据多样性
- 涌现能力（Emergent Ability）
- Transformer的模式匹配能力

### Q6: Encoder-Decoder vs Decoder-only的选择？
**答**：
- **Encoder-Decoder**（T5、BART）：
  - 适合需要理解全文的生成任务
  - 翻译、摘要效果好
- **Decoder-only**（GPT系列）：
  - 更简单、扩展性好
  - 统一的自回归生成
  - 当前主流（ChatGPT、GPT-4）

---

## 7️⃣ 关键技术细节

### Position Encoding
**BERT/GPT使用的绝对位置编码**：
```python
PE(pos, 2i) = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

**RoPE（旋转位置编码）**：
- LLaMA、GLM使用
- 相对位置信息
- 支持外推到更长序列

### Attention Mask
**BERT（双向）**：
```
Q: 我 爱 NLP
   ↓  ↓  ↓
   所有词都能看到所有词
```

**GPT（因果）**：
```
我  →  我
爱  →  我 爱
NLP →  我 爱 NLP
```

### Tokenization
- **WordPiece**（BERT）：子词分割
- **BPE**（GPT-2/3）：字节对编码
- **SentencePiece**（T5）：统一中英文处理

---

## 8️⃣ 实战技巧

### 选择合适的预训练模型
- **文本分类/NER**：BERT、RoBERTa
- **文本生成/对话**：GPT系列
- **翻译/摘要**：T5、BART
- **多任务**：T5
- **代码**：CodeBERT、GPT-3.5+

### 微调策略
- **全量微调**：所有参数都更新
- **LoRA**：只训练低秩矩阵
- **Prompt Tuning**：只优化prompt embedding
- **Adapter**：插入小模块

### 常见问题排查
- **过拟合**：减小学习率、增加dropout、使用更多数据
- **灾难性遗忘**：混合预训练数据、使用LoRA
- **训练不稳定**：梯度裁剪、warm-up、使用FP16

---

## 📚 参考论文

- BERT: https://arxiv.org/abs/1810.04805
- GPT: https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf
- GPT-2: https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
- GPT-3: https://arxiv.org/abs/2005.14165
- T5: https://arxiv.org/abs/1910.10683
- RoBERTa: https://arxiv.org/abs/1907.11692
- ALBERT: https://arxiv.org/abs/1909.11942
