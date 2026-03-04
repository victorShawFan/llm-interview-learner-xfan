# 大模型预训练

## 预训练目标

### Causal Language Modeling (CLM)
- 预测下一个token：P(x_t | x_1, ..., x_{t-1})
- GPT系列使用
- 单向注意力

### Masked Language Modeling (MLM)  
- 随机mask 15%tokens预测
- BERT使用
- 双向注意力

## 预训练数据

### 数据来源
- CommonCrawl：网页爬虫数据
- Books：图书语料
- Wikipedia
- Code：GitHub代码
- 总量：数TB原始文本

### 数据清洗
- 去重：MinHash/SimHash
- 过滤低质量：困惑度过滤
- 有害内容过滤
- 多语言平衡

## 训练技巧

### 学习率调度
- Warmup + Cosine Decay
- Warmup防止训练初期梯度爆炸
- 最佳：warmup 2000步，然后decay

### 混合精度训练
- FP16/BF16计算，FP32累积梯度
- 加速2x，显存减半
- 需要Loss Scaling防止下溢

### 梯度累积
- 小batch多步累积等效大batch
- 公式：effective_batch = batch * accum_steps * gpus

## 分布式训练

### Data Parallelism (DP)
- 每个GPU一份完整模型
- 数据切分，梯度AllReduce
- 简单但显存占用大

### ZeRO (Zero Redundancy Optimizer)
- Stage 1：切分optimizer states
- Stage 2：切分gradients  
- Stage 3：切分parameters
- 显存优化，通信开销增加

### 训练稳定性
- Gradient Clipping：防梯度爆炸
- 监控loss spike
- checkpoint恢复机制
