# 大模型微调技术

## 为什么需要微调？

预训练大模型（如GPT、LLaMA）具备通用能力，但在特定领域或任务上表现可能不佳：
- 垂直领域知识不足（医疗、法律、金融专业术语）
- 指令遵循能力弱（回答格式不规范、理解意图偏差）
- 输出风格不符（需要正式/口语化/特定语气）
- 安全对齐不够（可能生成有害/偏见内容）

微调通过在特定数据上继续训练，使模型适应下游任务。

## 微调的主要类型

### 1. 全量微调（Full Fine-Tuning）

更新模型所有参数

**方法**：
- 在标注数据上继续训练，使用小学习率（如1e-5）
- 损失函数：交叉熵（语言建模）或特定任务loss

**优势**：
- 效果最好，模型完全适配任务
- 可以学习全新领域知识

**劣势**：
- 显存需求大：70B模型FP16需要140GB显存（仅权重）
- 训练慢：需要更新数十亿参数
- 容易灾难性遗忘：微调后通用能力下降
- 需要为每个任务保存完整模型副本

**适用场景**：
- 有充足资源（A100 x8）
- 任务与预训练差异大（如中文模型微调为日语）
- 需要最佳性能

### 2. 参数高效微调（PEFT）

只更新少量参数，冻结大部分权重

**核心思想**：大模型是过参数化的，只需微调少量参数即可适配任务

**主流方法对比**：

| 方法 | 可训练参数 | 显存占用 | 推理成本 | 效果 |
|------|------------|----------|----------|------|
| 全量微调 | 100% | 极高 | 不变 | 最优 |
| LoRA | 0.1-1% | 低 | 微增 | 接近全量 |
| Adapter | 1-5% | 中等 | 增加 | 较好 |
| Prefix Tuning | 0.1% | 低 | 不变 | 中等 |
| Prompt Tuning | <0.01% | 极低 | 不变 | 一般 |

## LoRA（Low-Rank Adaptation）

当前最流行的PEFT方法

### 核心原理

假设权重更新 $\Delta W$ 是低秩的，可以分解为两个小矩阵：

$$
W' = W + \Delta W = W + A \cdot B
$$

- $W \in \mathbb{R}^{d \times k}$：原始权重（冻结）
- $A \in \mathbb{R}^{d \times r}$, $B \in \mathbb{R}^{r \times k}$：可训练的低秩矩阵
- $r \ll \min(d, k)$：秩（如$r=8$）

### 参数量对比

以LLaMA-7B为例，对Query、Value矩阵应用LoRA（$r=16$）：
- 原始参数：7B
- LoRA参数：$2 \times 32 \times 4096 \times 16 \times 2 \approx 8.4M$（0.12%）
- 显存占用：从28GB降至4GB

### 实现

```python
import torch
import torch.nn as nn

class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank=16, alpha=16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        
        self.lora_A = nn.Parameter(torch.randn(in_dim, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_dim))
        
    def forward(self, x, original_weight):
        lora_output = x @ self.lora_A @ self.lora_B * (self.alpha / self.rank)
        original_output = x @ original_weight
        return original_output + lora_output
```

### LoRA变体

**（1）QLoRA（Quantized LoRA）**
- 基座模型量化为4-bit（NF4格式）
- LoRA权重保持FP16
- 显存需求再降50%
- 论文：65B模型在单张A100 48GB上微调

**（2）AdaLoRA**
- 动态调整不同层的秩$r$
- 重要层分配更大秩，不重要层秩小
- 参数利用更高效

**（3）LoRA+**
- 优化学习率：$A$矩阵用小LR，$B$矩阵用大LR
- 收敛更快，效果更好

### LoRA最佳实践

**哪些层应用LoRA？**
- Query、Value矩阵：必选（影响最大）
- Key矩阵：可选
- FFN（MLP）：可选，增加参数量但提升效果
- 经验：只Q/V已够好，全部应用效果最佳但参数多

**秩$r$如何选择？**
- $r=8$：极低资源，效果尚可
- $r=16$：常用默认值，性价比高
- $r=64$：高精度任务，接近全量微调
- 任务越难、数据越多，$r$应越大

**$\alpha$如何设置？**
- 常见：$\alpha = r$或$\alpha = 2r$
- 作用：缩放LoRA更新的强度
- $\alpha/r$大 → LoRA影响大，学习激进
- $\alpha/r$小 → LoRA影响小，稳定但慢

## Adapter

在Transformer层中插入小模块

### 结构

```
Input
  ↓
LayerNorm
  ↓
Self-Attention (冻结)
  ↓
+───→ Adapter Module (可训练)
  ↓
LayerNorm
  ↓
FFN (冻结)
  ↓
+───→ Adapter Module (可训练)
  ↓
Output
```

**Adapter Module**：
```
Down-projection: d → bottleneck (如d/16)
Activation: ReLU/GELU
Up-projection: bottleneck → d
Residual: 输出 + 输入
```

### 优劣

**优势**：
- 实现简单直观
- 可针对不同任务插入不同Adapter
- 推理时可动态切换Adapter

**劣势**：
- 增加推理延迟（额外前向传播）
- 参数比LoRA多（1-5% vs 0.1-1%）

## 指令微调（Instruction Tuning）

让模型学会遵循指令

### 数据格式

```json
{
  "instruction": "将以下句子翻译成英文",
  "input": "今天天气很好",
  "output": "The weather is nice today."
}
```

或对话格式：
```json
{
  "conversations": [
    {"from": "human", "value": "什么是量子纠缠？"},
    {"from": "gpt", "value": "量子纠缠是指两个或多个粒子..."}
  ]
}
```

### 训练目标

**Causal Language Modeling Loss**：
- 只对"output"部分计算loss
- "instruction" + "input"部分mask掉

```python
loss = CrossEntropy(logits[output_start:], labels[output_start:])
```

### 数据构造

**开源指令数据集**：
- Alpaca：52K指令（GPT-3.5生成）
- ShareGPT：真实ChatGPT对话
- BELLE：中文指令数据
- Open-Orca：复杂推理指令

**自己构造**：
- 用GPT-4生成：给定任务描述，让GPT-4生成指令-回答对
- 人工标注：针对特定领域，雇佣专家标注
- 合成数据：用规则或模板生成

### Instruction Tuning vs Supervised Fine-Tuning

- SFT：传统监督学习，直接在标注数据上训练
- Instruction Tuning：SFT的一种，强调"指令-输出"格式，提升泛化能力

## 对齐微调（RLHF / DPO）

使模型输出符合人类偏好

### RLHF（Reinforcement Learning from Human Feedback）

**三阶段流程**：

**1. Supervised Fine-Tuning (SFT)**
- 在高质量指令数据上微调，学会基本对话能力

**2. Reward Model (RM) 训练**
- 收集人类偏好数据：对同一prompt，模型生成多个回答，人类排序
- 训练奖励模型：输入(prompt, response)，输出分数
- 目标：$\text{score}(y_{\text{win}}) > \text{score}(y_{\text{lose}})$

**3. PPO强化学习**
- 用RM作为奖励函数，用PPO算法优化策略模型
- 目标：$\max_{\theta} \mathbb{E}_{x,y \sim \pi_\theta}[r(x,y) - \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}]$
  - $r(x,y)$：奖励模型打分
  - $\beta$：KL惩罚系数，防止偏离原模型太远

**挑战**：
- 训练复杂：需要4个模型（策略、奖励、参考、价值）
- 不稳定：RL训练容易崩溃
- 成本高：需要大量人类标注

### DPO（Direct Preference Optimization）

无需RL，直接从偏好数据优化

**核心思想**：绕过奖励模型，直接建模偏好

**损失函数**：
$$
\mathcal{L} = -\mathbb{E}_{(x,y_w,y_l)} \left[\log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]
$$

- $y_w$：人类偏好的回答
- $y_l$：人类不喜欢的回答
- 直接最大化$\pi_\theta$在$y_w$上的概率，降低$y_l$概率

**优势**：
- 无需训练RM
- 无需RL，直接监督学习
- 稳定性好，易调参
- 效果与RLHF相当甚至更好

**实践**：
- Llama 2、Mistral等模型都用DPO替代RLHF

## 领域微调实践

### 医疗领域

**数据**：
- 医疗问答：医生回答患者问题
- 病历分析：诊断报告、治疗方案
- 医学知识：教科书、文献

**挑战**：
- 专业性强，需医学专家标注
- 安全要求高，不能误诊
- 隐私问题：患者数据脱敏

**案例**：
- HuatuoGPT：中文医疗LLM，在ChatGLM基础上微调
- Med-PaLM：Google医疗模型，通过多轮RLHF对齐

### 代码生成

**数据**：
- Code-Instruct：自然语言→代码
- Code-Review：代码审查、修复Bug
- Code-Explain：解释代码功能

**技巧**：
- 执行反馈：模型生成代码后运行，根据是否通过测试调整
- Self-Instruct：用模型自己生成更多代码任务

**案例**：
- WizardCoder：StarCoder + Evol-Instruct微调
- CodeLlama：LLaMA专门微调代码版本

### 多轮对话

**数据格式**：
```json
{
  "conversations": [
    {"from": "human", "value": "你好"},
    {"from": "assistant", "value": "你好！有什么可以帮您？"},
    {"from": "human", "value": "推荐一部科幻电影"},
    {"from": "assistant", "value": "《星际穿越》很不错..."}
  ]
}
```

**训练技巧**：
- Packing：将多个短对话拼接，提升GPU利用率
- Position Embedding重置：每个对话独立计算位置

## 面试高频题

### Q1：全量微调为什么容易"灾难性遗忘"？如何缓解？

**答案要点**：

**原因**：
- 微调数据分布与预训练差异大，模型参数剧烈调整
- 小数据集过拟合，覆盖了预训练学到的通用知识
- 例如：在医疗数据上全量微调后，模型可能忘记如何写诗、翻译

**缓解方法**：
- **小学习率**：$10^{-5}$ ~ $10^{-6}$，温柔调整参数
- **更少epoch**：1-3个epoch，避免过拟合
- **混合数据**：微调时混入通用数据（如10%预训练数据）
- **正则化**：加L2正则，惩罚参数偏离预训练值太远
- **PEFT**：只微调小部分参数（如LoRA），保留预训练知识

### Q2：LoRA为什么$r$设小（如8）就够了？低秩假设的依据是什么？

**答案要点**：

**低秩假设**：
- 微调的权重更新$\Delta W$相比原权重$W$，维度较低
- 直觉：微调只是"微调"，不是从头学习，只需调整少量方向

**实验证据**：
- 论文《LoRA》测试发现，$r=8$时性能已接近全量微调
- 增大$r$到64，提升有限（边际递减）
- 说明权重更新确实集中在少数几个主方向

**理论解释**：
- 大模型过参数化，参数空间冗余
- 适配任务时，只需在低维子空间调整
- 类比：飞机已经造好（预训练），微调只是调整飞行参数，不需要重新设计引擎

### Q3：DPO相比RLHF的优势在哪？为什么能"无需RL"？

**答案要点**：

**RLHF问题**：
- 需要单独训练奖励模型（RM）
- RL训练（PPO）复杂、不稳定
- 需要维护4个模型，显存和计算成本高

**DPO关键洞察**：
- 奖励模型本质是对偏好建模：$r(x,y) = \log \pi^*(y|x) - \log \pi_{\text{ref}}(y|x)$
- 可以直接用策略模型$\pi_\theta$建模这个关系，绕过RM
- 将RL目标重参数化为监督学习目标

**优势**：
- 稳定：监督学习比RL稳定
- 简单：只需训练1个模型
- 高效：无需多次采样、奖励估计
- 效果：实验证明与RLHF相当甚至更好

**直觉**：
- RLHF是"间接"：先学奖励函数，再优化策略
- DPO是"直接"：从偏好数据直接学策略
- 类比：RLHF像考试先学评分标准再答题，DPO直接看范文学习
