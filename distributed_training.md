# LLM分布式训练

## 1. 为什么需要分布式训练？

**问题规模**：
- GPT-3: 1750亿参数，需要350GB显存（FP32）
- PaLM: 5400亿参数，需要超过1TB显存
- GPT-4: 推测1.8万亿参数（未公开）

**单卡限制**：
- A100 80GB: 无法容纳百亿级模型
- 训练时间：单卡训练GPT-3需要数百年

**解决方案**：分布式训练技术

---

## 2. 核心并行策略

### 2.1 数据并行（Data Parallelism, DP）

**原理**：每个GPU持有完整模型副本，输入数据切分到不同GPU。

**流程**：
```
GPU 0: Model Copy → Batch 0-7   → Gradients → AllReduce
GPU 1: Model Copy → Batch 8-15  → Gradients → AllReduce
GPU 2: Model Copy → Batch 16-23 → Gradients → AllReduce
GPU 3: Model Copy → Batch 24-31 → Gradients → AllReduce
                              ↓
                        梯度平均并同步
                              ↓
                        参数更新（一致）
```

**实现**（PyTorch DDP）：
```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# 初始化进程组
dist.init_process_group("nccl")

model = YourModel().cuda()
model = DDP(model, device_ids=[local_rank])

# 训练循环
for data, target in dataloader:
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()  # 自动AllReduce梯度
    optimizer.step()
```

**优缺点**：
- ✅ 简单易用，线性加速（理想情况）
- ✅ 适合小模型（<10B参数）
- ❌ 每张卡都存完整模型，显存浪费
- ❌ 通信开销大（AllReduce需同步所有梯度）

---

### 2.2 模型并行（Model Parallelism, MP）

#### 2.2.1 张量并行（Tensor Parallelism, TP）

**原理**：将单个层的参数切分到多个GPU。

**示例**（线性层切分）：
```
原始：Y = XW (X: [B, D_in], W: [D_in, D_out])

张量并行（按列切分W）：
GPU 0: W_0 [D_in, D_out/2]  →  Y_0 = X @ W_0
GPU 1: W_1 [D_in, D_out/2]  →  Y_1 = X @ W_1
最终: Y = [Y_0, Y_1]  # 拼接
```

**Megatron-LM的实现**：
- Self-Attention：QKV投影矩阵按列切分
- MLP：第一个FC按列切分，第二个FC按行切分

**代码示例**：
```python
class ColumnParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, world_size):
        super().__init__()
        assert out_features % world_size == 0
        self.out_features_per_gpu = out_features // world_size
        self.weight = nn.Parameter(
            torch.empty(in_features, self.out_features_per_gpu)
        )
    
    def forward(self, x):
        output = F.linear(x, self.weight)  # 局部计算
        return output  # 各GPU持有输出的一部分
```

**优缺点**：
- ✅ 突破单GPU显存限制
- ✅ 适合超宽层（如MLP hidden_size=40960）
- ❌ 通信频繁（每层都需要AllReduce）
- ❌ 需要高带宽网络（NVLink/InfiniBand）

---

#### 2.2.2 流水线并行（Pipeline Parallelism, PP）

**原理**：将模型按层切分，不同GPU负责不同层。

**Naive Pipeline（低效）**：
```
GPU 0: Layer 0-5   → Forward → Idle → Idle → Backward
GPU 1: Layer 6-11  → Idle → Forward → Idle → Backward
GPU 2: Layer 12-17 → Idle → Idle → Forward → Backward
```
问题：GPU利用率低（bubble time大）

**GPipe（改进）**：将batch切成micro-batch，流水线执行
```
时间步：  1   2   3   4   5   6   7   8
GPU 0:  F0  F1  F2  F3  B0  B1  B2  B3
GPU 1:  -   F0  F1  F2  F3  B0  B1  B2
GPU 2:  -   -   F0  F1  F2  F3  B0  B1
```
F=Forward, B=Backward

**代码框架**（DeepSpeed Pipeline）：
```python
from deepspeed.pipe import PipelineModule, LayerSpec

layers = [
    LayerSpec(TransformerLayer, hidden_size=1024),
    LayerSpec(TransformerLayer, hidden_size=1024),
    ...  # 24层
]

model = PipelineModule(
    layers=layers,
    num_stages=4,  # 4个GPU
    partition_method='uniform'
)
```

**优缺点**：
- ✅ 显存占用低（每GPU只存部分层）
- ✅ 通信少（只在边界层通信）
- ❌ Bubble time导致GPU闲置
- ❌ Micro-batch切分需调优

---

### 2.3 3D并行（DP + TP + PP）

**综合策略**：结合三种并行，训练超大模型。

**示例配置**（GPT-3 175B）：
```
总GPU数：1024
- 数据并行：16 (DP)
- 张量并行：8 (TP)
- 流水线并行：8 (PP)

每个TP组：8张GPU（高带宽互联）
每个PP阶段：处理 175B / 8 = 22B参数
数据并行组：16份数据副本
```

**通信拓扑**：
```
DP维度（跨节点）：低频大量数据
  ↓
TP维度（节点内NVLink）：高频小数据
  ↓
PP维度（跨节点）：中频中量数据
```

---

## 3. ZeRO优化器（微软DeepSpeed）

### 3.1 ZeRO核心思想

**问题**：数据并行中每张卡存储完整模型副本浪费显存。

**ZeRO-1**：切分优化器状态
- Adam优化器需要存：参数 + 梯度 + Momentum + Variance
- ZeRO-1：将Momentum和Variance切分到不同GPU

**ZeRO-2**：切分优化器状态 + 梯度
- 每张卡只存自己负责的那部分梯度

**ZeRO-3**：切分优化器状态 + 梯度 + 模型参数
- 每张卡只存模型的1/N，需要时通过AllGather获取

**显存对比**（以7B模型为例）：
| 方法 | 模型参数 | 梯度 | 优化器状态 | 总显存 |
|------|---------|------|-----------|--------|
| 原始DP | 28GB | 28GB | 56GB | 112GB |
| ZeRO-1 | 28GB | 28GB | 56GB/N | 56+56/N GB |
| ZeRO-2 | 28GB | 28GB/N | 56GB/N | 28+84/N GB |
| ZeRO-3 | 28GB/N | 28GB/N | 56GB/N | 112/N GB |

---

### 3.2 ZeRO实战配置

**DeepSpeed配置文件**：
```json
{
  "train_batch_size": 32,
  "gradient_accumulation_steps": 4,
  
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    },
    "overlap_comm": true,
    "contiguous_gradients": true,
    "reduce_bucket_size": 5e8,
    "stage3_prefetch_bucket_size": 5e8,
    "stage3_param_persistence_threshold": 1e6
  },
  
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "loss_scale_window": 1000,
    "hysteresis": 2,
    "min_loss_scale": 1
  }
}
```

**Python代码**：
```python
import deepspeed

model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config="ds_config.json"
)

for step, batch in enumerate(dataloader):
    loss = model_engine(batch)
    model_engine.backward(loss)
    model_engine.step()
```

---

### 3.3 ZeRO-Offload

**核心**：将优化器状态和梯度offload到CPU内存/NVMe磁盘。

**适用场景**：
- GPU显存不足但CPU内存充裕
- 牺牲少量速度换更大模型

**性能对比**（训练100B模型）：
| 方法 | GPU显存 | 训练速度 |
|------|---------|---------|
| ZeRO-3 | 32GB × 16 | 100% |
| ZeRO-Offload | 32GB × 8 | 75% |

---

## 4. 通信优化

### 4.1 梯度累积（Gradient Accumulation）

**问题**：大batch size显存放不下。

**解决**：分多个micro-batch累积梯度，再更新参数。

```python
accumulation_steps = 8
optimizer.zero_grad()

for i, (data, target) in enumerate(dataloader):
    output = model(data)
    loss = criterion(output, target) / accumulation_steps
    loss.backward()  # 梯度累积
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**等价关系**：
```
micro_batch=4, accumulation=8  ==  effective_batch=32
```

---

### 4.2 混合精度训练（Mixed Precision）

**FP16好处**：
- 显存占用减半
- 计算速度提升（Tensor Core加速）

**数值稳定性问题**：
- FP16表示范围小，易溢出
- 梯度太小会underflow为0

**解决方案**（NVIDIA Apex）：
```python
from apex import amp

model, optimizer = amp.initialize(model, optimizer, opt_level="O2")

for data, target in dataloader:
    output = model(data)
    loss = criterion(output, target)
    
    with amp.scale_loss(loss, optimizer) as scaled_loss:
        scaled_loss.backward()
    
    optimizer.step()
```

**O0-O3优化等级**：
- O0: FP32训练
- O1: 部分算子FP16（推荐）
- O2: 几乎全FP16，保留BN为FP32
- O3: 全FP16（可能不稳定）

---

### 4.3 通信后端选择

| 后端 | 适用场景 | 带宽 | 延迟 |
|------|---------|------|------|
| **NCCL** | GPU间通信（推荐） | 600GB/s (NVLink) | 低 |
| **Gloo** | CPU通信 | 10GB/s (以太网) | 中 |
| **MPI** | HPC集群 | 100GB/s (InfiniBand) | 低 |

---

## 5. 实战案例

### 案例1：训练13B模型（单机8卡A100）

**配置**：
```json
{
  "zero_optimization": {"stage": 2},
  "train_batch_size": 128,
  "gradient_accumulation_steps": 16,
  "fp16": {"enabled": true}
}
```

**结果**：
- 每GPU显存占用：45GB
- 训练吞吐：12K tokens/s
- 完整训练时间：1.5万亿tokens，约30天

---

### 案例2：训练175B模型（64节点，512张A100）

**策略**：
- DP=64 (跨节点)
- TP=4 (节点内NVLink)
- PP=2 (减少通信)
- ZeRO-1 (切分优化器)

**关键参数**：
```python
# Megatron-DeepSpeed配置
--tensor-model-parallel-size 4
--pipeline-model-parallel-size 2
--num-layers 96
--hidden-size 12288
--num-attention-heads 96
--seq-length 2048
--global-batch-size 1536
--micro-batch-size 1
```

**性能指标**：
- 每节点吞吐：150 TFLOPS
- 模型FLOPS利用率：52%（行业顶尖）
- 训练时间：3000亿tokens，约2周

---

## 6. 常见面试题

### Q1：数据并行和模型并行有什么区别？

**答案**：

| 维度 | 数据并行 (DP) | 模型并行 (MP) |
|------|--------------|--------------|
| **切分对象** | 输入数据 | 模型参数 |
| **显存占用** | 每卡完整模型 | 每卡部分模型 |
| **通信模式** | AllReduce梯度 | 层间激活传递 |
| **适用场景** | 模型小，数据大 | 模型大，单卡放不下 |
| **加速比** | 线性（理想） | 次线性（通信开销大） |

**实际应用**：
- <10B模型：纯DP
- 10B-100B：DP + TP
- >100B：DP + TP + PP (3D并行)

---

### Q2：ZeRO-2和ZeRO-3的核心区别是什么？

**答案**：

**ZeRO-2**：
- 切分：优化器状态 + 梯度
- 保留：完整模型参数在每张卡
- 通信：backward时ReduceScatter梯度

**ZeRO-3**：
- 切分：优化器状态 + 梯度 + 模型参数
- 每张卡只存1/N参数
- 通信：forward时AllGather参数，backward后释放

**选择建议**：
- ZeRO-2：通信少，速度快，但显存节省有限
- ZeRO-3：显存节省极致，但通信开销大（适合低带宽网络）

---

### Q3：为什么大模型训练需要混合精度？

**答案**：

**显存收益**：
- FP32: 4 bytes/param
- FP16: 2 bytes/param
- 175B模型：700GB → 350GB

**计算加速**：
- A100 Tensor Core: FP16是FP32的3倍吞吐

**数值稳定性保障**：
- Loss Scaling：梯度乘以scale防止underflow
- Master Weights：参数更新在FP32进行
- 动态调整scale：避免overflow

**实测效果**（GPT-3训练）：
- 纯FP32：基准
- 混合精度：速度提升2.5x，显存节省50%

---

### Q4：如何选择最优的并行策略？

**答案**（来源：腾讯云2025年分布式训练指南）：

**决策树**：
```
模型大小 < 10B？
  ├─ 是 → 纯数据并行（DDP + AMP）
  └─ 否 → 单卡能放下模型？
      ├─ 是 → ZeRO-2 + DP
      └─ 否 → 需要3D并行
          ├─ TP维度：8-16（节点内）
          ├─ PP维度：2-4（减少bubble）
          └─ DP维度：剩余GPU数

带宽情况？
  ├─ NVLink内存在 → TP优先
  ├─ InfiniBand → PP + DP
  └─ 以太网 → ZeRO-3 + DP
```

**经验公式**（Google Pathways论文）：
```
理想TP数 = min(8, 模型层宽度 / 单GPU显存)
理想PP数 = 2 或 4（避免bubble过大）
DP数 = 总GPU数 / (TP数 × PP数)
```

---

### Q5：什么是Gradient Checkpointing？

**答案**：

**问题**：
- Transformer训练需存储每层的激活值用于backward
- 96层模型，激活值占用大量显存

**解决**：
- Forward时只存部分层的激活（如每4层存1层）
- Backward时重新计算丢弃的激活

**Trade-off**：
- 显存节省：50-70%
- 计算时间增加：20-30%

**代码**：
```python
from torch.utils.checkpoint import checkpoint

class TransformerLayer(nn.Module):
    def forward(self, x):
        # 使用checkpoint包装计算密集型模块
        x = checkpoint(self.attention, x)
        x = checkpoint(self.mlp, x)
        return x
```

**实测（GPT-2 1.5B）**：
- 不用checkpoint：显存48GB
- 用checkpoint：显存28GB
- 训练速度：100% → 82%

---

## 7. 工具对比

| 框架 | 开发方 | 核心特性 | 适用规模 |
|------|--------|---------|---------|
| **DeepSpeed** | 微软 | ZeRO、3D并行、易用 | 10B-1000B+ |
| **Megatron-LM** | NVIDIA | TP/PP极致性能 | 100B-1000B |
| **FSDP** | Meta | PyTorch原生，ZeRO变体 | 10B-100B |
| **Colossal-AI** | HPC-AI | 中文友好，易上手 | 1B-100B |
| **Alpa** | UC Berkeley | 自动并行策略搜索 | 研究原型 |

---

## 8. 学习路径

### 初级（1-2周）
- [ ] 理解DP/TP/PP基本原理
- [ ] 实践PyTorch DDP训练小模型
- [ ] 掌握混合精度训练

### 中级（3-4周）
- [ ] 使用DeepSpeed训练10B+模型
- [ ] 对比ZeRO-2 vs ZeRO-3效果
- [ ] 学习Megatron-LM的TP实现

### 高级（5-6周）
- [ ] 设计自定义3D并行策略
- [ ] 优化通信拓扑和带宽利用
- [ ] 研究最新论文（如FlashAttention-2对通信的影响）

---

## 参考资料

- [DeepSpeed官方文档](https://www.deepspeed.ai/)
- [Megatron-LM GitHub](https://github.com/NVIDIA/Megatron-LM)
- [ZeRO论文](https://arxiv.org/abs/1910.02054) - ZeRO: Memory Optimizations Toward Training Trillion Parameter Models
- 腾讯云：LLM训练的高效分布式策略（2025）
- CSDN：DeepSpeed超大LLM分布式训练框架
