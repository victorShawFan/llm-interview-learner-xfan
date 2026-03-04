# Prompt Engineering 提示词工程

## 1. 核心概念

**Prompt Engineering（提示词工程）** 是通过精心设计输入文本（Prompt）来引导大语言模型（LLM）生成符合预期输出的技术。这是与LLM交互的核心技能，直接决定模型输出的质量、准确性和相关性。

### 为什么需要Prompt Engineering？

**LLM的特性**：
- 模型本质是"基于已生成token + 输入上下文预测下一个token"（Google 2025白皮书）
- 同一个任务，不同的Prompt可能导致输出质量天壤之别
- 好的Prompt可以激发模型的reasoning能力，差的Prompt会导致幻觉或答非所问

**示例对比**：
```
❌ 差Prompt：
"翻译这句话"

✅ 好Prompt：
"作为专业翻译，将以下英文句子翻译成地道的中文，保持原文的语气和风格：
[句子]
翻译："
```

---

## 2. Prompt的基本结构

一个完整的Prompt通常包含以下组成部分：

```
【角色定义】(Role)
你是一位资深的Python工程师

【任务描述】(Task)
请帮我优化以下代码的性能

【上下文信息】(Context)
这段代码用于处理百万级数据，当前运行时间10秒

【输入内容】(Input)
[代码片段]

【输出格式】(Output Format)
请按以下格式输出：
1. 性能瓶颈分析
2. 优化建议
3. 优化后的代码

【约束条件】(Constraints)
- 不改变原有功能
- 保持代码可读性
```

---

## 3. 核心技术

### 3.1 Zero-Shot Prompting（零样本）

**定义**：直接给出任务描述，不提供任何示例。

**适用场景**：简单任务、通用能力强的大模型（如GPT-4）。

**示例**：
```
Prompt: 将以下句子翻译成法语：
"The weather is nice today."

Output: "Le temps est beau aujourd'hui."
```

**优点**：简单快速
**缺点**：对复杂任务效果有限

---

### 3.2 Few-Shot Prompting（少样本学习）

**定义**：在Prompt中提供2-5个示例，引导模型学习任务模式。

**示例**：
```
Prompt:
请根据情感分类句子：

示例1：
句子："这部电影太好看了！"
情感：正面

示例2：
句子："服务态度很差，不推荐。"
情感：负面

示例3：
句子："还行吧，没什么特别的。"
情感：中性

现在请分类：
句子："产品质量超出预期，非常满意！"
情感：
```

**技巧**：
- 示例要有代表性，覆盖不同情况
- 通常3-5个示例效果最佳（太多会占用上下文窗口）
- 示例顺序会影响结果，建议按难度递增排列

---

### 3.3 Chain-of-Thought (CoT) 思维链

**定义**：要求模型逐步展示推理过程，而不是直接给出答案。

**关键短语**：`"Let's think step by step"` `"一步一步分析"`

**示例**：
```
❌ 直接提问：
Q: Roger有5个网球。他又买了2罐网球，每罐3个球。他现在有多少网球？
A: 11个

✅ CoT提问：
Q: Roger有5个网球。他又买了2罐网球，每罐3个球。他现在有多少网球？
请一步一步思考：

A: 让我们一步步分析：
1. Roger原本有5个网球
2. 他买了2罐，每罐3个球，所以买了 2 × 3 = 6个球
3. 总共：5 + 6 = 11个网球
答案：11个
```

**进阶技术**：
- **Zero-Shot CoT**：只需加一句"Let's think step by step"即可激活推理
- **Self-Consistency**：多次采样，选择最一致的答案

---

### 3.4 Tree of Thoughts (ToT) 思维树

**定义**：扩展CoT，探索多条推理路径，像树一样展开搜索空间。

**适用场景**：复杂推理、策略规划、创意生成。

**示例**（24点游戏）：
```
给定数字：4, 9, 10, 13
目标：通过 +, -, ×, ÷ 得到24

思维树展开：
路径1：(13-9)×(10-4) = 4×6 = 24 ✅
路径2：(10-4)×(13-9) = 6×4 = 24 ✅
路径3：13×9-10×4 = 117-40 = 77 ❌
...
```

---

### 3.5 Self-Consistency（自洽性）

**定义**：多次采样生成，选择出现最频繁的答案。

**流程**：
1. 用相同的CoT Prompt生成10次
2. 统计每个答案的频率
3. 选择出现最多的答案

**代码示例**：
```python
from collections import Counter

answers = []
for _ in range(10):
    response = llm.generate(prompt)
    answers.append(extract_answer(response))

most_common = Counter(answers).most_common(1)[0][0]
print(f"最终答案：{most_common}")
```

---

### 3.6 ReAct（Reasoning + Acting）

**定义**：交替进行推理（Thought）和行动（Action），常用于Agent。

**格式**：
```
Question: 2023年诺贝尔物理学奖得主是谁？

Thought 1: 我需要搜索最新的诺贝尔奖信息
Action 1: Search["2023年诺贝尔物理学奖"]
Observation 1: Pierre Agostini, Ferenc Krausz, Anne L'Huillier

Thought 2: 找到了三位获奖者，我需要确认他们的贡献
Action 2: Search["阿秒激光脉冲"]
Observation 2: 用于研究物质中的电子动力学

Thought 3: 已经收集到足够信息
Action 3: Finish[Pierre Agostini, Ferenc Krausz, Anne L'Huillier因阿秒激光脉冲研究获奖]
```

---

## 4. 进阶技巧

### 4.1 角色扮演（Role Prompting）

给模型赋予特定角色，激活相关领域知识。

**示例**：
```
你是一位拥有20年经验的心理咨询师，擅长认知行为疗法。
请为以下来访者提供专业建议：
[来访者描述]
```

**常用角色**：
- 领域专家：`资深Python工程师` `经济学教授` `法律顾问`
- 性格特点：`严谨的逻辑学家` `富有创意的文案` `耐心的老师`

---

### 4.2 分隔符与格式化

**使用分隔符明确区分内容**：
```
### 任务描述 ###
总结以下文章

### 文章内容 ###
[长文本]

### 输出要求 ###
1. 不超过100字
2. 保留关键信息
```

**结构化输出（JSON/Markdown）**：
```
请以JSON格式输出分析结果：
{
  "sentiment": "positive/negative/neutral",
  "confidence": 0.0-1.0,
  "keywords": ["关键词1", "关键词2"]
}
```

---

### 4.3 负面提示（Negative Prompting）

明确告诉模型**不要做什么**。

**示例**：
```
请生成一篇产品介绍：
要求：
✅ 突出产品优势
✅ 使用数据支撑
❌ 不要使用夸张的营销语言
❌ 避免与竞品比较
❌ 不要虚构用户评价
```

---

### 4.4 思维引导模板

**ICIO框架**（Input-Context-Instruction-Output）：
```
[Input] 用户评论："这个手机电池不耐用"
[Context] 这是一条产品差评
[Instruction] 请生成客服回复，体现同理心并提供解决方案
[Output Format] 控制在50字以内
```

**RASCEF框架**（Role-Action-Steps-Context-Examples-Format）：
```
Role: 你是产品经理
Action: 撰写需求文档
Steps: 1.背景分析 2.用户痛点 3.解决方案 4.预期效果
Context: 针对电商App的购物车功能
Examples: [提供1-2个示例文档]
Format: Markdown格式，包含标题、列表、表格
```

---

## 5. 常见陷阱与解决方案

### 陷阱1：提示词过于模糊

**问题**：
```
❌ "写一篇文章"
```

**解决**：
```
✅ "写一篇800字的科普文章，主题是'量子计算机的工作原理'，
面向高中生读者，使用通俗易懂的语言，包含2-3个实际应用案例"
```

---

### 陷阱2：忽视上下文长度限制

**问题**：一次性输入超长文档，导致截断或OOM。

**解决**：
- 分块处理（Chunking）
- MapReduce模式：先分别总结每个chunk，再合并总结
- 使用长上下文模型（如Claude 200K）

---

### 陷阱3：指令冲突

**问题**：
```
❌ "详细解释每个步骤，但控制在50字以内"
（详细 vs 简短 矛盾）
```

**解决**：
```
✅ "用列表形式列出关键步骤（3-5步），每步10字以内"
```

---

### 陷阱4：过度依赖Few-Shot示例

**问题**：示例质量差或有偏见，模型会学到错误模式。

**解决**：
- 精选高质量示例
- 确保示例多样性
- 必要时用Zero-Shot + CoT替代

---

## 6. 评估Prompt质量

### 6.1 定量指标

| 指标 | 计算方法 | 适用场景 |
|------|---------|---------|
| **准确率** | 正确答案数 / 总数 | 分类、QA |
| **BLEU** | n-gram重叠度 | 翻译、摘要 |
| **ROUGE** | 召回率为主 | 摘要生成 |
| **Perplexity** | 困惑度（越低越好）| 文本生成 |

---

### 6.2 定性评估

- **相关性**：输出是否回答了问题
- **连贯性**：逻辑是否清晰
- **事实性**：是否符合客观事实（无幻觉）
- **安全性**：是否包含有害内容

---

### 6.3 A/B测试

对比两个Prompt的效果：
```python
prompt_A = "总结以下文章：{text}"
prompt_B = "请用3句话总结以下文章的核心观点：{text}"

# 对100篇文章进行测试
scores_A = evaluate(prompt_A, test_set)
scores_B = evaluate(prompt_B, test_set)

print(f"Prompt A平均分：{np.mean(scores_A)}")
print(f"Prompt B平均分：{np.mean(scores_B)}")
```

---

## 7. 实战案例

### 案例1：代码生成

**任务**：生成Python函数实现快速排序。

**优化前**：
```
写一个快速排序
```

**优化后**：
```
作为Python专家，请编写一个快速排序函数，要求：
1. 函数名：quick_sort
2. 输入：整数列表
3. 输出：排序后的列表
4. 包含详细注释
5. 时间复杂度分析

示例：
输入：[3, 1, 4, 1, 5, 9, 2, 6]
输出：[1, 1, 2, 3, 4, 5, 6, 9]
```

---

### 案例2：SQL查询生成

**任务**：根据自然语言生成SQL。

**Prompt模板**：
```
数据库Schema：
- users表：id, name, email, created_at
- orders表：id, user_id, product_id, amount, order_date
- products表：id, name, price, category

请根据以下需求生成SQL查询：
"找出2023年购买金额超过1000元的前10名用户及其购买总额"

要求：
1. 使用标准SQL语法
2. 包含必要的JOIN
3. 添加注释说明关键步骤
```

---

### 案例3：情感分析

**Few-Shot + CoT结合**：
```
请分析以下评论的情感倾向（正面/负面/中性）并说明理由：

示例1：
评论："外卖送达很快，但菜品温度不够"
分析：虽然配送速度被肯定，但菜品质量存在问题，整体偏负面
情感：负面

示例2：
评论："价格合理，性价比不错"
分析：用户对价格和性价比表示满意
情感：正面

现在请分析：
评论："包装精美，但口味一般般"
分析：
```

---

## 8. 常见面试题

### Q1：什么是Prompt Engineering？它为什么重要？

**答案**：
Prompt Engineering是通过精心设计输入文本（Prompt）来引导大语言模型生成符合预期输出的技术。

**重要性**：
1. **提升输出质量**：好的Prompt可以将准确率从60%提升到90%+
2. **降低成本**：减少重复调用次数，节省API费用
3. **激发模型能力**：CoT等技术可以激发模型的推理能力
4. **无需Fine-tuning**：通过Prompt调整即可适配不同任务

---

### Q2：Zero-Shot、One-Shot、Few-Shot有什么区别？

**答案**：

| 类型 | 示例数量 | 适用场景 | 优缺点 |
|------|---------|---------|--------|
| **Zero-Shot** | 0个 | 简单任务、通用模型 | 简单但效果有限 |
| **One-Shot** | 1个 | 任务模式明确 | 容易过拟合到示例 |
| **Few-Shot** | 2-5个 | 复杂任务、小样本学习 | 效果好但占用上下文 |

**示例**：
```
Zero-Shot: "翻译成中文：Hello"

Few-Shot: 
"翻译示例：
Hello -> 你好
Goodbye -> 再见
现在翻译：How are you?"
```

---

### Q3：什么是Chain-of-Thought (CoT)？如何使用？

**答案**：
CoT是要求模型逐步展示推理过程的技术，通过中间步骤提升复杂推理任务的准确性。

**使用方法**：
1. **手动CoT**：在示例中展示完整推理步骤
2. **Zero-Shot CoT**：添加"Let's think step by step"即可

**效果提升**：
- GSM8K（数学题）：Zero-Shot 17% → CoT 40%
- CommonsenseQA：62% → 79%

---

### Q4：如何防止LLM产生幻觉（Hallucination）？

**答案**：

**Prompt层面**：
1. **明确要求引用来源**：`"请基于给定文档回答，不要编造信息"`
2. **要求自我验证**：`"请检查你的答案是否有事实错误"`
3. **使用RAG**：结合检索，基于真实文档生成

**技术手段**：
- **Self-Consistency**：多次采样，选择一致性高的答案
- **External Verification**：调用搜索引擎或知识库验证
- **Uncertainty Estimation**：让模型输出置信度

---

### Q5：如何设计Prompt让模型输出结构化数据（如JSON）？

**答案**：

**方法1：明确输出格式**
```
请以JSON格式输出，schema如下：
{
  "name": "字符串",
  "age": "整数",
  "skills": ["技能1", "技能2"]
}
```

**方法2：使用Function Calling**（OpenAI）
```python
functions = [{
    "name": "extract_info",
    "parameters": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        }
    }
}]
```

**方法3：Few-Shot示例**
```
示例1：
输入："张三，25岁，Python工程师"
输出：{"name":"张三","age":25,"job":"Python工程师"}

现在处理：
输入："李四，30岁，产品经理"
输出：
```

---

### Q6：Prompt Engineering与Fine-tuning有什么区别？

**答案**：

| 维度 | Prompt Engineering | Fine-tuning |
|------|-------------------|-------------|
| **修改对象** | 输入文本 | 模型参数 |
| **成本** | 低（无需训练） | 高（需要GPU） |
| **速度** | 快（实时调整） | 慢（需重新训练） |
| **灵活性** | 高（随时修改） | 低（固化到模型中） |
| **数据需求** | 少（0-100条示例） | 多（1000+标注数据） |
| **适用场景** | 快速迭代、多任务 | 固定任务、极致性能 |

**最佳实践**：
- 先用Prompt Engineering快速验证
- 效果稳定后考虑Fine-tuning进一步优化

---

### Q7：什么是Temperature和Top-p采样？如何调整？

**答案**：

**Temperature（温度参数）**：
- 范围：0-2（通常0-1）
- 作用：控制输出的随机性
- 低温（0.1-0.3）：输出确定性强，适合事实性任务（QA、翻译）
- 高温（0.7-1.0）：输出多样性高，适合创意任务（故事、头脑风暴）

**Top-p（核采样）**：
- 范围：0-1
- 作用：只从累计概率达到p的token中采样
- 低p（0.1-0.5）：保守输出
- 高p（0.9-0.95）：多样输出

**推荐配置**：
```python
# 代码生成
{"temperature": 0.2, "top_p": 0.95}

# 创意写作
{"temperature": 0.8, "top_p": 0.95}

# 严格事实性任务
{"temperature": 0, "top_p": 1.0}  # Greedy Decoding
```

---

## 9. 实战工具推荐

### 9.1 Prompt优化工具

- **PromptPerfect**：自动优化Prompt
- **LangChain PromptTemplate**：Prompt模板管理
- **OpenAI Playground**：可视化调试
- **PromptBase**：Prompt交易市场

---

### 9.2 评估框架

```python
from langchain.evaluation import load_evaluator

evaluator = load_evaluator("criteria", criteria="conciseness")
result = evaluator.evaluate_strings(
    prediction="模型输出",
    input="原始Prompt"
)
```

---

## 10. 学习路径

### 初级（1-2周）
- [ ] 理解Zero-Shot、Few-Shot、CoT基本概念
- [ ] 练习编写清晰的任务描述
- [ ] 尝试不同角色定义的效果

### 中级（3-4周）
- [ ] 掌握ReAct、Self-Consistency等进阶技术
- [ ] 学习结构化输出（JSON、Markdown）
- [ ] 对比不同Prompt的A/B测试

### 高级（5-6周）
- [ ] 设计领域特定的Prompt模板库
- [ ] 研究Prompt Injection攻击与防御
- [ ] 优化Token使用效率

---

## 参考资料

- [Google Prompt Engineering白皮书（2025）](https://ai.google.dev/docs/prompt_best_practices)
- [OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering)
- [Anthropic Prompt Engineering](https://docs.anthropic.com/claude/docs/prompt-engineering)
- 掘金：AI大模型岗位面试题之 Prompt 提示词工程
- B站：2025最全提示词工程教程
- 今日头条：写Prompt不再靠玄学
