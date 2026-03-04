# LLM评估与基准测试

## 1. 为什么需要评估LLM？

**挑战**：
- 如何判断模型好坏？
- 不同模型如何对比？
- 优化后效果如何量化？

**评估维度**：
1. **能力评估**：语言理解、推理、知识
2. **安全评估**：有害内容、偏见、隐私
3. **效率评估**：推理速度、成本

---

## 2. 主流基准测试

### 2.1 通用能力基准

**MMLU（Massive Multitask Language Understanding）**
- 57个学科，15,908道选择题
- 涵盖STEM、人文、社科
- GPT-4: 86.4%, Claude 3: 86.8%

**GLUE/SuperGLUE**
- 语言理解任务集合
- 情感分析、自然语言推理、问答

**BIG-Bench**
- 204个多样化任务
- 测试创造力、推理、常识

---

### 2.2 中文基准

**C-Eval**
- 52个学科，13,948道题
- 针对中文模型优化

**CMMLU**
- 中国文化、历史、法律专项测试

---

### 2.3 代码能力

**HumanEval**
- 164道Python编程题
- GPT-4: 67%, Claude 3.5: 92%

**MBPP**
- 974道基础编程题

---

### 2.4 数学推理

**GSM8K**
- 小学数学应用题
- 8,500道题

**MATH**
- 高中竞赛数学
- 12,500道题

---

## 3. 评估方法

### 3.1 自动化评估

**准确率（Accuracy）**
```python
correct = sum([pred == label for pred, label in zip(predictions, labels)])
accuracy = correct / len(labels)
```

**BLEU/ROUGE（生成任务）**
- 机器翻译、文本摘要

---

### 3.2 人工评估

**Elo评分（Chatbot Arena）**
- 用户投票对比模型
- GPT-4、Claude动态排名

**A/B测试**
- 真实用户场景对比

---

## 4. 常见面试题

### Q1：如何评估LLM的代码生成能力？

**答案**：
1. **HumanEval基准**：164道编程题，执行通过率
2. **功能正确性**：单元测试覆盖
3. **代码质量**：可读性、注释、命名规范
4. **效率**：时间/空间复杂度

---

### Q2：MMLU和C-Eval有什么区别？

**答案**：
- MMLU：英文，西方教育体系
- C-Eval：中文，中国教育体系
- C-Eval更关注中国文化、政治、法律

---

### Q3：为什么不能只用准确率评估？

**答案**：
- 忽略生成质量（流畅度、相关性）
- 无法评估创造性任务
- 需要结合人工评估、用户满意度

---

## 5. 学习资源

- [MMLU论文](https://arxiv.org/abs/2009.03300)
- [C-Eval官网](https://cevalbenchmark.com/)
- [HumanEval数据集](https://github.com/openai/human-eval)
