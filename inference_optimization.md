# 大模型推理优化技术

## 推理性能指标

- **延迟（Latency）**：生成第一个token的时间（TTFT）和整体响应时间
- **吞吐量（Throughput）**：单位时间处理的tokens数或请求数
- **显存占用（Memory）**：推理时GPU显存消耗
- **成本（Cost）**：GPU使用成本

## KV Cache优化

### 问题
Transformer推理时，每生成一个token都需要用到之前所有token的Key和Value：
- 每个token的KV: `2 * layers * hidden_dim * 2bytes`
- 2048 tokens, LLaMA-70B: 约40GB显存

### PagedAttention (vLLM)
- 将KV cache分页存储（类似OS虚拟内存）
- 页大小如64 tokens
- 优势：减少内存碎片，提升batch吞吐23x

### Multi-Query Attention (MQA)
- 所有Query head共享一组K和V
- KV cache减少为原来的1/32（假设32个head）
- 问题：精度略降

### Grouped-Query Attention (GQA)
- MQA和MHA折中：Query分组，每组共享KV
- 8组GQA：KV cache为MHA的1/4
- LLaMA 2 70B使用GQA

## 量化推理

### INT8量化
- 权重和激活量化为INT8
- 推理速度提升2x，显存减半
- LLM.int8()：混合精度，敏感层用FP16

### INT4/AWQ
- 权重量化为4-bit
- 显存降至1/4
- AWQ：根据激活重要性量化，保持精度

### GPTQ
- 后训练量化，针对生成式模型优化
- 4-bit量化，精度损失<1%

## 推测解码（Speculative Decoding）

### 原理
用小模型预测多个token，大模型并行验证：
1. 小模型自回归生成K个tokens
2. 大模型一次前向验证K个tokens
3. 接受正确的前N个，丢弃后续
4. 重复

### 效果
- 加速2-3x（取决于小模型准确率）
- 无精度损失（输出分布一致）
- 适合大小模型配合使用

## Continuous Batching

### 问题
传统batching：等最长序列生成完，浪费GPU

### 解决
动态调整batch：
- 请求完成立即移出batch
- 新请求立即加入batch
- Iteration-level scheduling

### 实现
- vLLM、TensorRT-LLM支持
- 吞吐量提升10x+

## 模型并行推理

### Tensor Parallelism
- 权重矩阵按列或行切分到多GPU
- 适合大模型单机多卡
- 通信开销大

### Pipeline Parallelism
- 按层切分，不同GPU负责不同层
- Micro-batch流水线执行
- 通信少但存在气泡

## Flash-Decoding

优化解码阶段attention计算：
- Prefill阶段：一次计算所有input tokens（并行）
- Decode阶段：每次只生成1个token（串行瓶颈）
- Flash-Decoding：优化Decode的attention kernel
- 加速解码4-8x

## 面试题

### Q1：为什么推理阶段KV cache是瓶颈？
- 生成长文本时KV cache占主要显存
- 每个token都需读取完整KV cache
- 内存带宽受限（memory-bound）

### Q2：Speculative Decoding为何无损？
- 大模型最终决定接受哪些token
- 只是用小模型提前猜测，加速并行验证
- 输出概率分布与原始大模型一致

### Q3：vLLM相比HuggingFace Transformers快在哪？
- PagedAttention：减少显存碎片和浪费
- Continuous Batching：动态调度
- Kernel优化：CUDA kernel手写优化
- 吞吐量提升10-20x
