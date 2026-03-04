# LLM 推理优化技术

## 🚀 为什么需要推理优化？

大语言模型推理面临的挑战：
- **显存占用大**：70B模型需要140GB显存（FP16）
- **推理速度慢**：逐token生成，延迟高
- **成本高**：GPU资源昂贵
- **吞吐量低**：batch size受限

---

## 1️⃣ 模型压缩技术

### 1.1 量化（Quantization）

#### FP16/BF16（半精度）
```python
# FP32 → FP16
model = model.half()  # PyTorch
model = model.to(torch.float16)
```
- 内存减半：32bit → 16bit
- 速度提升：现代GPU对FP16优化好
- 精度损失：很小，几乎无感

**BF16 vs FP16**：
| 格式 | 符号位 | 指数位 | 尾数位 | 特点 |
|------|--------|--------|--------|------|
| FP32 | 1 | 8 | 23 | 标准精度 |
| FP16 | 1 | 5 | 10 | 易溢出 |
| BF16 | 1 | 8 | 7 | 动态范围大，精度略低 |

**选择**：
- 训练：BF16（防止溢出）
- 推理：FP16/BF16都可

#### INT8量化
```python
# Post-Training Quantization
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

model = AutoModelForCausalLM.from_pretrained(
    "model_name",
    load_in_8bit=True,  # INT8量化
    device_map="auto"
)
```

**原理**：
```python
# 对称量化
scale = max(|W|) / 127
W_int8 = round(W / scale)

# 反量化
W_fp = W_int8 * scale
```

**优势**：
- 内存减少75%（32bit → 8bit）
- 精度损失：1-2% perplexity增加
- 推理速度提升：2-3倍

#### INT4量化（GPTQ、AWQ）
```python
# GPTQ量化
model = AutoGPTQForCausalLM.from_quantized(
    "model_name",
    use_safetensors=True,
    device_map="auto"
)
```

**GPTQ**：
- Group-wise量化
- 每128个元素共享scale
- 4bit存储，推理时解压到FP16

**AWQ（Activation-aware Weight Quantization）**：
- 根据激活值重要性量化
- 保护重要权重
- 精度更高

**对比**：
| 方法 | 精度 | 速度 | 实现难度 |
|------|------|------|----------|
| GPTQ | 较好 | 快 | 中 |
| AWQ | 更好 | 较快 | 较高 |
| QLoRA | 最好 | 中 | 低（训练时） |

### 1.2 剪枝（Pruning）

#### 非结构化剪枝
- 移除权重矩阵中接近0的元素
- 稀疏矩阵存储
- 需要硬件支持

#### 结构化剪枝
- 移除整个注意力头或FFN层
- 不改变模型结构
- 兼容性好

```python
# 示例：移除注意力头
def prune_attention_heads(model, heads_to_prune):
    """
    heads_to_prune: {layer_id: [head_1, head_2, ...]}
    """
    for layer_idx, heads in heads_to_prune.items():
        layer = model.encoder.layer[layer_idx]
        layer.attention.prune_heads(heads)
```

### 1.3 知识蒸馏（Knowledge Distillation）

**原理**：大模型（Teacher）指导小模型（Student）

```python
# 蒸馏loss
loss = α * CE(y_student, y_true) + (1-α) * KL(y_student, y_teacher)
```

**DistilBERT示例**：
- BERT-base（110M） → DistilBERT（66M）
- 保留97%性能
- 速度提升60%

---

## 2️⃣ 推理加速技术

### 2.1 KV-Cache

**问题**：自回归生成每步都要重新计算历史token的K、V

**解决**：缓存已计算的K、V矩阵

```python
class KVCache:
    def __init__(self):
        self.keys = []    # 缓存所有历史K
        self.values = []  # 缓存所有历史V
    
    def update(self, new_key, new_value):
        self.keys.append(new_key)
        self.values.append(new_value)
        return torch.cat(self.keys, dim=1), torch.cat(self.values, dim=1)
```

**效果**：
- 时间：O(n²) → O(n)
- 显存：增加，但可接受
- 必备优化

### 2.2 Flash Attention

**核心**：优化GPU内存访问模式

**传统Attention**：
```python
# 需要存储n×n的attention矩阵
S = Q @ K.T / sqrt(d)  # n×n矩阵
P = softmax(S)         # n×n矩阵
O = P @ V
```

**Flash Attention**：
- 分块计算（Tiling）
- 利用SRAM（快速缓存）
- 不存储完整attention矩阵
- IO次数：O(n²) → O(n²/M), M为SRAM大小

```python
# PyTorch 2.0内置
from torch.nn.functional import scaled_dot_product_attention

output = scaled_dot_product_attention(
    query, key, value,
    attn_mask=None,
    is_causal=True  # 自动使用Flash Attention
)
```

**性能提升**：
- 速度：2-4倍
- 显存：3-5倍减少
- 长序列尤其明显

### 2.3 PagedAttention（vLLM）

**问题**：KV-Cache需要连续内存，利用率低

**解决**：分页存储KV-Cache

```
传统：
[████████████████____]  利用率60%，碎片化

PagedAttention：
[████][████][████]     分页存储，利用率90%+
```

**优势**：
- 吞吐量提升2-4倍
- 支持更大batch size
- 显存利用率大幅提升

**vLLM示例**：
```python
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-2-7b-hf")
prompts = ["Hello, my name is", "The capital of France is"]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

outputs = llm.generate(prompts, sampling_params)
```

### 2.4 Speculative Decoding（推测解码）

**核心思想**：小模型快速生成候选，大模型并行验证

```
1. 小模型生成k个token（快）
2. 大模型并行验证这k个token（一次forward）
3. 接受正确的，拒绝错误的
4. 从第一个错误处继续
```

**加速比**：
- 理论：2-3倍
- 实际：取决于小模型准确率
- 无精度损失（输出与大模型相同）

---

## 3️⃣ 批处理优化

### 3.1 Dynamic Batching

**问题**：不同请求生成长度不同，短的等长的

**解决**：动态组batch，完成的请求立即返回

```python
class DynamicBatcher:
    def __init__(self, max_batch_size=32, max_wait_time=0.1):
        self.queue = []
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
    
    def add_request(self, request):
        self.queue.append(request)
        
        if len(self.queue) >= self.max_batch_size:
            return self.process_batch()
    
    def process_batch(self):
        batch = self.queue[:self.max_batch_size]
        self.queue = self.queue[self.max_batch_size:]
        return model.generate(batch)
```

### 3.2 Continuous Batching（连续批处理）

**创新**：新请求可以加入正在推理的batch

```
传统：
Batch1: [完成] → Batch2开始
等待时间长

连续批处理：
Batch: [请求1完成，立即加入请求4]
吞吐量高
```

**实现**：vLLM、TensorRT-LLM

---

## 4️⃣ 模型架构优化

### 4.1 Multi-Query Attention（MQA）

**问题**：每个head都有独立的K、V矩阵

**改进**：所有head共享K、V

```
传统MHA：
Q: h个头，每个头d_k维
K: h个头，每个头d_k维  ← 独立
V: h个头，每个头d_v维  ← 独立

MQA：
Q: h个头，每个头d_k维
K: 1组，d_k维  ← 共享
V: 1组，d_v维  ← 共享
```

**优势**：
- KV-Cache减少h倍
- 推理速度提升
- 精度略有下降

**应用**：PaLM、Falcon

### 4.2 Grouped-Query Attention（GQA）

**折中方案**：K、V不完全共享，分组共享

```
8个Query头 → 2组KV（每组4个Q头共享）
```

**优势**：
- 平衡性能和精度
- Llama 2使用

### 4.3 Sparse Attention

**动机**：不是所有token都需要attend所有token

**方法**：
- **局部窗口**：只attend附近token
- **全局token**：少数token attend全局
- **随机采样**：随机选择部分token

**Longformer模式**：
```
局部窗口 + 全局token
█████░░░░░
░█████░░░░
░░█████░░░
█░░█████░█
█░░░█████░
```

---

## 5️⃣ 硬件与部署优化

### 5.1 张量并行（Tensor Parallelism）

**切分方式**：按层内切分

```python
# 权重矩阵按列切分
W = [W1 | W2 | W3 | W4]  # 4个GPU

# 前向传播
Y = X @ W
  = X @ [W1 | W2 | W3 | W4]
  = [Y1 | Y2 | Y3 | Y4]  # 拼接
```

**适用**：单模型大于单GPU显存

### 5.2 流水线并行（Pipeline Parallelism）

**切分方式**：按层间切分

```
GPU1: Layer 1-8
GPU2: Layer 9-16
GPU3: Layer 17-24
GPU4: Layer 25-32
```

**Micro-batch**：减少气泡时间

### 5.3 混合精度推理

```python
# PyTorch AMP
with torch.cuda.amp.autocast():
    output = model(input)
```

**策略**：
- 大部分计算用FP16
- 累加用FP32（防止精度损失）
- 自动混合

---

## 6️⃣ 推理框架对比

| 框架 | 特点 | 适用场景 |
|------|------|----------|
| **vLLM** | PagedAttention、连续批处理 | 高吞吐量服务 |
| **TensorRT-LLM** | NVIDIA优化、极致性能 | 生产环境 |
| **Text Generation Inference** | HuggingFace官方、易用 | 快速部署 |
| **DeepSpeed-Inference** | 微软出品、易集成训练 | 研究 |
| **llama.cpp** | CPU推理、量化 | 边缘设备 |

### vLLM示例

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    tensor_parallel_size=2,  # 2张GPU
    max_num_seqs=256         # 最大并发数
)

prompts = ["Hello"] * 100
outputs = llm.generate(prompts)
```

---

## 7️⃣ 面试高频问题

### Q1: 如何选择量化方法？
**答**：
- **精度优先**：AWQ、QLoRA
- **速度优先**：GPTQ
- **易用性优先**：BitsAndBytes（HuggingFace集成）
- **极致压缩**：INT4 + GPTQ

### Q2: KV-Cache的内存占用如何计算？
**答**：
```
KV-Cache大小 = 2 × layers × batch_size × seq_len × hidden_size × bytes

示例（Llama-7B，FP16）：
= 2 × 32 × 1 × 2048 × 4096 × 2
= 1GB（单个序列2048 tokens）
```

### Q3: Flash Attention为什么快？
**答**：
- 减少HBM（显存）访问次数
- 利用SRAM（L2 cache）
- 分块计算，不存储完整attention矩阵
- IO复杂度降低

### Q4: vLLM的PagedAttention原理？
**答**：
- KV-Cache按页存储（如4KB/页）
- 页可以不连续
- 动态分配和回收
- 显存利用率从60%提升到90%+

### Q5: 推理优化的优先级？
**答**：
1. **KV-Cache**（必备）
2. **量化**（FP16/INT8，性价比高）
3. **Flash Attention**（长序列必备）
4. **批处理优化**（提升吞吐）
5. **模型并行**（超大模型）

---

## 8️⃣ 实战建议

### 开发阶段
1. 使用FP16
2. 小batch size快速迭代
3. HuggingFace Transformers即可

### 生产部署
1. vLLM或TensorRT-LLM
2. INT8量化
3. 动态批处理
4. 监控显存和延迟

### 性能调优
```python
# 性能分析
import torch.profiler as profiler

with profiler.profile(
    activities=[profiler.ProfilerActivity.CPU,
                profiler.ProfilerActivity.CUDA],
    with_stack=True
) as prof:
    output = model(input)

print(prof.key_averages().table())
```

---

## 📚 参考资源

- Flash Attention: https://arxiv.org/abs/2205.14135
- vLLM: https://arxiv.org/abs/2309.06180
- GPTQ: https://arxiv.org/abs/2210.17323
- AWQ: https://arxiv.org/abs/2306.00978
- Speculative Decoding: https://arxiv.org/abs/2211.17192
