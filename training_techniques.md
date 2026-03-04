# LLM训练技术

## 🎓 训练范式演进

```
传统监督学习
    ↓
预训练-微调（BERT时代）
    ↓
Prompt Learning（GPT-3）
    ↓
指令微调 + RLHF（ChatGPT）
    ↓
DPO、ORPO等新方法
```

---

## 1️⃣ 预训练（Pre-training）

### 1.1 训练目标

#### 语言模型（CLM - Causal Language Modeling）
```python
# GPT系列使用
loss = CrossEntropy(model(x[:-1]), x[1:])
# 预测下一个词
```

**优势**：
- 简单有效
- 适合生成任务
- 可扩展性强

#### MLM（Masked Language Modeling）
```python
# BERT使用
masked_x = mask_tokens(x, mask_prob=0.15)
loss = CrossEntropy(model(masked_x), x)
# 预测被mask的词
```

### 1.2 数据配方（Data Recipe）

**Llama 2数据构成**：
- CommonCrawl网页：67%
- C4（清洗后）：15%
- GitHub代码：4.5%
- Wikipedia：4.5%
- Books：4.5%
- ArXiv论文：2.5%
- StackExchange：2%

**关键要素**：
- 数据量：几TB到几十TB
- 多样性：不同领域、语言、格式
- 质量：去重、过滤有害内容
- 混合比例：需要精心调整

### 1.3 训练技巧

#### 学习率调度
```python
# Warmup + Cosine Decay
def get_lr(step, warmup_steps, total_steps, max_lr, min_lr):
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return min_lr + (max_lr - min_lr) * 0.5 * (1 + cos(pi * progress))
```

**参数**：
- Warmup：1-10%的总步数
- Max LR：1e-4到6e-4（GPT-3用3e-4）
- Min LR：Max LR的10%

#### 梯度裁剪
```python
# 防止梯度爆炸
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

#### 混合精度训练
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    with autocast():
        loss = model(batch)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

---

## 2️⃣ 指令微调（Instruction Tuning）

### 2.1 数据格式

```json
{
  "instruction": "将下面的句子翻译成英文",
  "input": "今天天气真好",
  "output": "The weather is really nice today"
}
```

### 2.2 数据集

| 数据集 | 规模 | 特点 |
|--------|------|------|
| **FLAN** | 1800+ tasks | Google出品，多任务 |
| **P3** | 170+ datasets | Prompt模板丰富 |
| **Alpaca** | 52K | Self-Instruct生成 |
| **ShareGPT** | 90K | 真实对话数据 |
| **Dolly** | 15K | 人工标注 |

### 2.3 Self-Instruct

**流程**：
```
1. 种子任务（175个人工标注）
    ↓
2. GPT-3.5生成新任务
    ↓
3. GPT-3.5生成输入输出
    ↓
4. 过滤质量不佳的样本
    ↓
5. 用于微调
```

**优势**：
- 低成本获取大量数据
- 多样性好
- 快速迭代

---

## 3️⃣ RLHF（Reinforcement Learning from Human Feedback）

### 3.1 三阶段流程

#### Stage 1: SFT（Supervised Fine-Tuning）
```python
# 标准监督学习
loss = -log