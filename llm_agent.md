# LLM Agent智能体系统

## Agent是什么？

LLM Agent（智能体）是基于大语言模型的自主决策系统，能够：
- 感知环境（Perception）：理解用户输入和外部信息
- 制定计划（Planning）：分解任务为可执行步骤
- 采取行动（Action）：调用工具完成具体操作
- 反思改进（Reflection）：评估结果并调整策略

**Agent vs 简单LLM对话的区别**：
- 简单对话：用户问 → 模型答（单轮交互）
- Agent：用户目标 → 模型规划 → 调用工具 → 验证结果 → 循环迭代（多轮自主决策）

## Agent核心组件

### 1. 规划模块（Planning）

将复杂任务分解为子任务序列

**常见规划策略**：

**（1）ReAct（Reasoning + Acting）**
- 思考（Thought）→ 行动（Action）→ 观察（Observation）循环
- 示例：
  ```
  Thought: 用户想知道今天北京天气，我需要调用天气API
  Action: call_weather_api(city="北京", date="2024-03-04")
  Observation: {"temp": 15, "condition": "晴"}
  Thought: 已获取天气信息，现在可以回答用户
  Answer: 今天北京天气晴，温度15度
  ```

**（2）思维链（Chain-of-Thought, CoT）**
- 让模型先推理再行动，提升复杂任务准确率
- Zero-shot CoT：在Prompt加"Let's think step by step"
- Few-shot CoT：给几个推理示例

**（3）思维树（Tree of Thoughts, ToT）**
- 探索多个可能的推理路径，选择最优解
- 适合需要回溯的任务（如数学题、游戏）
- 使用BFS/DFS搜索思维空间

**（4）层次化规划（Hierarchical Planning）**
- 将任务分解为高层目标和低层动作
- 例如："写一篇论文" → "搜索文献" + "整理大纲" + "撰写章节"

### 2. 工具调用（Tool Use）

Agent通过Function Calling与外部世界交互

**工具类型**：
- 信息检索：搜索引擎、数据库查询、API调用
- 代码执行：Python解释器、沙盒环境
- 文件操作：读写文件、图像生成
- 外部服务：发邮件、订票、控制智能家居

**Function Calling实现**：
```python
# 定义工具schema
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名称"},
                    "date": {"type": "string", "description": "日期(YYYY-MM-DD)"}
                },
                "required": ["city"]
            }
        }
    }
]

# 模型决策调用工具
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "北京今天天气怎么样"}],
    tools=tools,
    tool_choice="auto"  # 让模型自主决定是否调用工具
)

# 模型输出：tool_calls=[{"function": {"name": "get_weather", "arguments": '{"city":"北京"}'}}]
```

**工具选择策略**：
- 单步工具：每次只调用一个工具
- 并行工具：同时调用多个独立工具（如同时查天气和股票）
- 链式工具：一个工具的输出作为下一个工具的输入

### 3. 记忆模块（Memory）

Agent需要记住历史信息来保持上下文连贯

**记忆类型**：

**（1）短期记忆（Working Memory）**
- 存储当前对话上下文（最近几轮对话）
- 实现：滑动窗口保留最近N条消息
- 挑战：上下文长度限制（如GPT-4 8K/32K tokens）

**（2）长期记忆（Long-term Memory）**
- 持久化存储重要信息
- 实现方式：
  - 向量数据库（如Pinecone、Chroma）：存储对话embeddings，相似度检索
  - 知识图谱：结构化存储实体关系
  - 总结压缩：定期总结历史对话，压缩为关键信息

**（3）情景记忆（Episodic Memory）**
- 记录Agent执行的具体任务和结果
- 用于从失败中学习："上次这个工具调用失败了，这次换个参数"

**记忆检索策略**：
- 最近优先：优先检索最近的对话
- 相似度检索：用embedding相似度找相关记忆
- 重要性加权：给重要信息（如用户偏好）更高权重

### 4. 反思模块（Reflection）

Agent评估自己的行动效果，并改进策略

**Reflexion框架**：
1. Agent尝试任务 → 失败
2. 自我反思："为什么失败？哪里可以改进？"
3. 生成反思总结（Reflection）存入记忆
4. 下次遇到类似任务时，参考反思改进策略

**Self-Refine**：
- Agent生成初步答案
- 自我批评："这个答案有什么问题？"
- 改进答案
- 重复直到满意

**实践技巧**：
- 让Agent输出置信度："我对这个答案80%确定"
- 多Agent互相评审：Agent A生成答案，Agent B批评，Agent A改进

## 主流Agent框架

### 1. AutoGPT
- 目标：全自主Agent，给定目标后自主规划和执行
- 特点：
  - 自主分解任务为子目标
  - 调用搜索、代码执行、文件读写等工具
  - 长期记忆（向量数据库）
- 局限：容易陷入循环、幻觉、成本高

### 2. LangChain Agent
- 提供Agent框架和工具生态
- Agent类型：
  - Zero-shot ReAct：根据工具描述动态选择
  - Conversational：带对话记忆
  - Self-ask with search：自问自答+搜索
- 工具生态：Google搜索、SQL、Python REPL、Wolfram Alpha等

```python
from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI

tools = [
    Tool(name="Search", func=google_search, description="用于搜索互联网"),
    Tool(name="Calculator", func=calculator, description="用于数学计算")
]

agent = initialize_agent(
    tools, 
    OpenAI(temperature=0), 
    agent="zero-shot-react-description"
)

agent.run("2023年世界杯冠军是谁？他们赢了多少场比赛？")
```

### 3. BabyAGI
- 任务驱动的自主Agent
- 工作流：
  1. 从任务队列拉取任务
  2. 执行任务并记录结果
  3. 根据目标和结果生成新任务
  4. 优先级排序任务队列

### 4. MetaGPT
- 多Agent协作框架，模拟软件公司
- 角色分工：
  - Product Manager：需求分析
  - Architect：系统设计
  - Engineer：编码实现
  - QA Engineer：测试
- 多Agent通过"文档"通信（PRD、设计文档、代码）

### 5. ChatDev
- 类似MetaGPT，模拟软件开发团队
- 瀑布式开发流程：设计 → 编码 → 测试 → 文档
- 多Agent辩论机制：CEO和CTO讨论方案直到达成一致

## Agent应用场景

### 1. 个人助理
- 日程管理：查日历、订会议室、发提醒
- 信息整合：收集邮件/新闻/社交媒体重要信息
- 自动化工作流：定期生成周报、整理文档

### 2. 代码助手
- 需求 → 设计 → 编码 → 测试全流程
- 自动修Bug：读代码 → 理解错误 → 生成修复 → 验证测试
- 代码审查：检查潜在问题、安全漏洞、性能优化点

### 3. 数据分析
- 自然语言查询数据库："过去一周销量前10的商品"
- 自动生成报表：写SQL → 查询 → 可视化 → 撰写分析
- 异常检测：发现数据异常 → 调查原因 → 生成报告

### 4. 游戏NPC
- 动态对话：根据玩家行为和游戏状态生成自然对话
- 自主行为：NPC根据目标（如"保护村庄"）自主决策
- 记忆系统：记住与玩家的互动历史

## Agent挑战与局限

### 1. 幻觉问题
- LLM生成不存在的信息（如编造API、捏造事实）
- 缓解方法：
  - 工具验证：调用搜索/数据库确认事实
  - 多模型投票：多个模型交叉验证
  - 人类在环（Human-in-the-loop）：关键决策需人确认

### 2. 无限循环
- Agent陷入重复动作（如反复搜索同一内容）
- 解决方案：
  - 最大迭代次数限制
  - 检测重复行为并强制切换策略
  - 引入随机性避免局部最优

### 3. 成本与延迟
- 多轮迭代导致API调用次数多、成本高
- 复杂推理链延迟大（几十秒到几分钟）
- 优化方向：
  - 使用更便宜的模型处理简单任务
  - 缓存常见查询结果
  - 并行化独立任务

### 4. 安全风险
- Agent可能执行危险操作（删文件、发垃圾邮件）
- 恶意Prompt注入：用户诱导Agent执行非预期行为
- 防护措施：
  - 沙盒执行环境
  - 危险操作需要人类确认
  - Prompt注入防护（如输入过滤）

## 面试高频题

### Q1：如何评估Agent系统的性能？

**答案要点**：
- 任务成功率：Agent完成任务的比例（最直接指标）
- 效率指标：
  - 平均步数：完成任务需要多少轮迭代
  - Token消耗：总API调用成本
  - 延迟：从接收任务到完成的时间
- 质量指标：
  - 答案准确率（与人类标注对比）
  - 幻觉率：生成不准确信息的比例
  - 工具调用准确性：是否调用了正确的工具
- Benchmark：
  - WebShop：电商购物任务
  - ALFWorld：文本游戏环境
  - HotPotQA：多跳问答

### Q2：为什么Agent需要ReAct而不是直接让LLM输出最终答案？

**答案要点**：
- 复杂推理需要分步：人类解决复杂问题时也是分步骤，每一步获取新信息后再决策，而不是一开始就知道所有答案
- 外部工具依赖：许多任务需要实时信息（天气、股票）或计算能力（Python执行），LLM内部知识无法覆盖
- 可解释性：ReAct显式输出思考过程，方便调试和理解Agent决策
- 纠错能力：如果中间步骤失败（如API调用报错），Agent可以观察错误并调整策略；一步到位则无法纠错
- 降低幻觉：通过工具获取真实信息，避免LLM编造

### Q3：多Agent系统相比单Agent有什么优势？什么情况下应该用多Agent？

**答案要点**：

**优势**：
- 专业化分工：每个Agent专注一个领域（如MetaGPT的PM/工程师/测试），类比人类团队
- 互相验证：多Agent互相批评和改进，减少错误（类似Code Review）
- 并行加速：独立任务可以由多Agent并行处理
- 模拟复杂系统：Agent之间博弈/竞争/合作，适合模拟社会、市场、游戏

**适用场景**：
- 复杂工程任务：软件开发需要需求分析、设计、编码、测试多个角色
- 辩论推理：多Agent从不同角度论证，提升推理质量
- 模拟仿真：模拟市场中的买家/卖家，社交网络中的用户

**不适用场景**：
- 简单任务：单Agent就能搞定，多Agent反而增加通信开销
- 成本敏感：多Agent意味着更多API调用
- 低延迟要求：多Agent串行通信增加延迟
