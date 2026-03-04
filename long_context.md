# 大模型长上下文技术

## 为什么需要长上下文？

传统LLM上下文长度受限（GPT-3.5: 4K tokens，GPT-4: 8K/32K），但实际应用需要更长上下文：
- 文档分析：处理百页PDF、论文、法律合同
- 代码理解：分析整个代码仓库（10万行+）
- 对话记忆：保持长时间对话上下文
- 检索增强（RAG）：处理大量检索结果

长上下文模型能处理100K甚至百万tokens，但面临技术挑战。

## 长上下文的核心挑战

### 1. 计算复杂度问题

Transformer的自注意力机制复杂度为O(n²)：
- 序列长度翻倍 → 计算量增加4倍
- 8K→128K：计算量增加256倍，显存消耗爆炸

**示例**：
- 4K tokens：16M计算量，2GB显存
- 128K tokens：16B计算量（1000倍），64GB显存

### 2. 注意力稀疏问题

实验发现：长文本中，模型只关注少数关键位置
- "Lost in the Middle"现象：模型对开头和结尾记得清楚，中间容易遗忘
- 大量位置的注意力权重接近0，浪费计算

### 3. 位置编码失效

标准位置编码（如sinusoidal）在超长序列上泛化能力差：
- 训练时见过的最长位置是4K，推理时遇到100K位置就失效
- 绝对位置编码无法表示超出训练范围的位置

## 主流长上下文技术

### 1. 高效注意力机制

**（1）FlashAttention**
- 思想：优化注意力计算的GPU内存访问模式
- 方法：
  - 分块计算attention，减少HBM（高带宽内存）访问
  - 利用GPU SRAM（快速缓存）存储中间结果
  - 精确计算，无损精度
- 效果：
  - 速度提升2-4x
  - 显存降低10-20x
  - 支持64K+ tokens
- 代码示例：
```python
from flash_attn import flash_attn_func

# 标准attention: O = softmax(QK^T / sqrt(d)) V
# FlashAttention: 分块计算，内存高效
output = flash_attn_func(q, k, v, causal=True)  # q,k,v: [batch, seqlen, heads, head_dim]
```

**（2）Sparse Attention（稀疏注意力）**
- 思想：不是所有token都需要关注所有token，只关注重要的
- 类型：
  - 局部窗口：每个token只关注前后W个token（如Longformer）
  - 全局token：少数token（如[CLS]）关注全文，其他token只局部关注
  - 随机采样：随机选择一部分token关注
- 复杂度：O(n) 或 O(n√n)
- 问题：可能丢失重要长距离依赖

**（3）线性注意力**
- 思想：用核方法近似softmax，将O(n²)降为O(n)
- 方法（如Linformer, Performer）：
  ```
  标准: Attention(Q,K,V) = softmax(QK^T)V  # O(n²)
  线性: Attention(Q,K,V) ≈ φ(Q)(φ(K)^T V)  # O(n), φ是核函数
  ```
- 问题：近似误差，效果不如精确attention

### 2. 位置编码改进

**（1）RoPE（Rotary Position Embedding）**
- 思想：通过旋转矩阵编码相对位置
- 优势：
  - 自然泛化到更长序列（外推性好）
  - 相对位置建模，长距离关系更稳定
- 公式：
  ```
  q_m = R_m q,  k_n = R_n k
  q_m^T k_n = q^T R_{m-n} k  # 只依赖相对位置m-n
  ```
- 外推技巧：
  - **线性插值**：位置m插值到m/k，缩小位置跨度
  - **NTK-aware插值**：高频低频分开处理

**（2）ALiBi（Attention with Linear Biases）**
- 思想：在attention score上加线性偏置，距离越远偏置越负
- 公式：
  ```
  score_{ij} = q_i^T k_j - λ·|i-j|  # λ是斜率超参数
  ```
- 优势：
  - 无需学习位置编码
  - 外推能力强（训练2K推理100K）
  - 实现简单

### 3. 上下文压缩

**（1）滑动窗口（Sliding Window）**
- 保留最近N个tokens，丢弃更早的
- 问题：丢失历史信息
- 改进：保留关键历史tokens（如问题、指令）

**（2）Memory Compression**
- 将早期tokens压缩为"记忆向量"
- 方法：
  - 用小模型总结早期内容
  - 用额外MLP压缩隐藏状态
- 示例：Memorizing Transformers保留最近4K，用kNN检索历史

**（3）检索增强（RAG + 长上下文）**
- 不把所有内容塞进上下文，先检索相关部分
- 流程：
  1. 将长文档切片，向量化存入数据库
  2. 用户查询时，检索top-k相关片段
  3. 将检索结果+查询输入LLM
- 优势：处理无限长文档，显存可控

### 4. 分层处理

**（1）Hierarchical Transformers**
- 低层：处理局部信息（每128 tokens一组）
- 高层：处理组间全局信息
- 复杂度：O(n²/k)，k是分组大小

**（2）Recursive Processing**
- 将长文本分段，每段单独处理
- 汇总各段结果，再做一次全局推理
- 适合摘要、问答等任务

## 主流长上下文模型

### 1. GPT-4 Turbo (128K)
- 方法：未公开，推测用FlashAttention + 改进位置编码
- 特点：
  - 128K context window
  - 支持300页PDF输入
  - 价格：$0.01/1K input tokens
- 局限：中间内容仍会"Lost in the Middle"

### 2. Claude 2.1 (200K)
- 方法：Anthropic自研注意力优化
- 特点：
  - 200K tokens（约150K words，500页书）
  - 检索能力强，中间内容也能较好召回
- 测试："大海捞针"实验，200K文本中找一句话，准确率90%+

### 3. Gemini 1.5 Pro (1M)
- 方法：
  - Sparse MoE（混合专家）+ 高效attention
  - 分层处理，不同层看不同粒度
- 特点：
  - 1M tokens（约70万words）
  - 能处理1小时视频，11小时音频
  - 甚至输入整个代码仓库
- 成本：推理慢，API昂贵

### 4. Kimi (200K)
- 月之暗面开源模型
- 方法：
  - FlashAttention-2
  - Grouped Query Attention（GQA）减少KV cache
  - 长度外推技巧
- 特点：
  - 国产长上下文标杆
  - 中文效果好
  - 价格便宜

### 5. LongLLaMA (256K)
- 基于LLaMA魔改
- 方法：
  - FoT（Focused Transformer）：动态选择关注范围
  - Memory cache存储关键信息
- 开源，社区可复现

## 长上下文的实际问题

### 1. "Lost in the Middle"

**现象**：即使模型支持128K，放在中间的信息仍容易被忽略

**实验**（Liu et al. 2023）：
- 在长文档的开头、中间、结尾插入答案
- 结果：开头准确率90%，中间60%，结尾85%

**原因**：
- 训练数据多为"问题在开头，答案在文中"格式
- 注意力分散，中间权重小

**缓解方法**：
- 重要信息放开头或结尾
- 用检索先筛选，再放入上下文
- Few-shot示例放在中间，引导模型关注中间

### 2. 显存墙

**KV Cache问题**：
- 推理时需要缓存所有历史token的Key、Value向量
- 显存消耗：`2 * batch * seq_len * layers * hidden_dim * 2bytes`
- 128K序列，LLaMA-70B需要140GB显存（仅KV cache）

**解决方案**：
- Grouped Query Attention（GQA）：多个Query共享KV，减少KV cache
- Paged Attention（vLLM）：KV cache分页管理，减少碎片
- Quantization：KV cache量化为INT8/INT4

### 3. 推理延迟

**问题**：序列越长，推理越慢
- 128K输入，首token延迟可达30秒
- 后续token生成受限于KV cache访问速度

**优化**：
- 预填充阶段并行：一次性计算所有input tokens的attention
- 解码阶段优化：用FlashDecoding等技术加速

### 4. 成本高昂

**API成本**：
- GPT-4 Turbo 128K：$0.01/1K input，$0.03/1K output
- 一次128K输入 + 2K输出 = $1.34
- 高频调用成本爆炸

**自建成本**：
- 需要A100 80GB或H100
- 电费、散热、维护成本

## 评估长上下文能力

### 1. 针（Needle）测试
- 在长文本中随机插入一句"针"信息
- 测试模型能否找到
- 评估不同位置的召回率

### 2. LongBench
- 多任务长文本Benchmark
- 包含：文档问答、代码理解、摘要等
- 序列长度覆盖4K-128K

### 3. 压力测试
- 逐渐增加输入长度，观察性能衰减
- 测试"外推"能力：训练4K，测试32K

## 面试高频题

### Q1：为什么不直接把Transformer的position encoding范围扩大到100K？

**答案要点**：
- 绝对位置编码（如Sinusoidal）虽然理论上可以计算任意位置，但模型在训练时只见过0-4K的位置，对超出范围的位置泛化能力差（就像外推，训练数据分布之外）
- 模型需要在长序列数据上重新训练或微调，才能适应新的位置分布
- 计算复杂度：即使位置编码支持100K，自注意力O(n²)计算量仍会爆炸，需要用FlashAttention等优化

**更好的方案**：
- 用相对位置编码（RoPE, ALiBi），天然支持外推
- 在长序列数据上继续训练（continual pretraining）

### Q2：FlashAttention如何做到加速且无损精度？

**答案要点**：
- 关键：优化GPU内存层次访问，而非改变计算逻辑
- 标准attention：Q, K, V从HBM读入，计算QK^T（巨大中间矩阵），再算softmax和乘V，频繁读写HBM（慢）
- FlashAttention：
  - 分块计算：将Q,K,V切成小块，每次只在SRAM（快）中计算一块
  - Tiling：外层循环遍历块，内层在SRAM做完整attention计算
  - Online softmax：增量更新softmax统计量，避免存储完整attention矩阵
- 结果：减少HBM访问次数（从O(n²)到O(n²/M)，M是SRAM大小），但计算逻辑完全一致，精度无损
- 类比：不是改变做饭步骤，而是优化厨房布局减少走动

### Q3：如何判断一个长上下文模型是"真长"还是"假长"？

**答案要点**：

**假长表现**：
- 只是位置编码支持长序列，但实际能力没提升
- 测试方法：
  - Needle测试：在不同位置插入信息，看召回率。假长模型中间位置召回率骤降
  - 性能曲线：随序列增长，perplexity快速上升
  - 实际任务：长文档问答准确率低

**真长表现**：
- 在长序列数据上充分训练
- 测试全面：不仅测位置外推，还测任务性能（问答、摘要、推理）
- 中间位置信息也能准确召回（"Needle in Haystack" 90%+）

**验证方法**：
- 看技术报告：是否在长序列数据（如书籍、长代码）上训练
- 看Benchmark：LongBench等多任务测试
- 看成本：真长上下文计算成本高，太便宜可能有猫腻（如sliding window fake长度）
