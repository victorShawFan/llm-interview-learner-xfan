# LLM力扣面试真题复习

## 1. 算法面试题整理
### 快速排序
```python
# 快速排序实现
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr)//2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)
```

### 二分查找
```python
# 二分查找实现
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
```

## 2. 深度学习与NLP核心知识点

### 2.1 Batch Normalization
- **定义**：对每个batch的数据进行归一化，使数据分布稳定
- **优点**：加速训练、减少梯度消失、允许更大的学习率
- **计算**：对mini-batch计算均值和方差，进行标准化
- **面试重点**：与Layer Norm的区别

### 2.2 Layer Normalization
- **定义**：对每个样本的特征维度进行归一化
- **应用场景**：RNN、Transformer（因为序列长度可变）
- **与BN的区别**：
  - BN沿batch维度归一化，LN沿特征维度归一化
  - BN依赖batch size，LN不依赖
  - Transformer使用Layer Norm

### 2.3 优化器对比：SGD vs Adam
- **SGD（随机梯度下降）**：
  - 简单但需要精心调整学习率
  - 可能陷入局部最优
  - 对学习率敏感
  
- **Adam（Adaptive Moment Estimation）**：
  - 自适应学习率
  - 结合了动量（Momentum）和RMSProp
  - 适合大多数深度学习任务
  - 更新公式：m_t = β1*m_(t-1) + (1-β1)*g_t, v_t = β2*v_(t-1) + (1-β2)*g_t^2

### 2.4 Attention机制 ⭐
- **核心思想**：让模型关注输入的重要部分
- **计算流程**：
  1. 计算Query和Key的相似度
  2. 对相似度做softmax得到权重
  3. 权重与Value加权求和
- **公式**：Attention(Q,K,V) = softmax(QK^T/√d_k)V
- **应用**：机器翻译、文本生成、图像描述

### 2.5 Multi-head Self-Attention
- **多头机制**：并行多个attention头，捕获不同子空间的信息
- **Self-Attention**：Q、K、V都来自同一个输入
- **优势**：
  - 捕获不同类型的依赖关系
  - 增强模型表达能力
  - 并行计算效率高
- **计算**：MultiHead(Q,K,V) = Concat(head_1,...,head_h)W^O

### 2.6 Dropout
- **定义**：训练时随机丢弃部分神经元（设为0）
- **目的**：防止过拟合，提高泛化能力
- **实现**：p概率保留神经元，输出需要乘1/p
- **使用场景**：全连接层、Attention层
- **注意**：测试时不使用dropout

### 2.7 Transformer架构 ⭐⭐⭐
- **核心**：完全基于Attention机制，抛弃了RNN/CNN
- **架构**：Encoder-Decoder结构
  - **Encoder**：输入 → Self-Attention → Add&Norm → FFN → Add&Norm → 输出
  - **Decoder**：类似但有Masked Self-Attention和Cross-Attention
  
- **Transformer Block结构**：
  ```
  Input
    ↓
  Multi-Head Self-Attention
    ↓
  Add & Layer Norm（残差连接）
    ↓
  Feed Forward Network（FFN）
    ↓
  Add & Layer Norm（残差连接）
    ↓
  Output
  ```

- **关键组件**：
  1. **位置编码（Positional Encoding）**：弥补无序列信息
  2. **残差连接（Residual Connection）**：缓解梯度消失
  3. **Layer Normalization**：稳定训练
  4. **前馈网络（FFN）**：两层全连接+ReLU

- **优势**：
  - 并行计算能力强
  - 捕获长距离依赖
  - 可解释性好（attention权重可视化）

- **应用**：BERT、GPT、T5等预训练模型的基础

## 3. LLM常见面试问题
1. LLM的训练过程是怎样的？
2. 如何解决过拟合问题？
3. 注意力机制的原理是什么？
4. 你对大模型的伦理问题有什么看法？
5. Transformer的核心创新是什么？
6. 为什么Transformer比RNN更适合处理长序列？
7. Layer Norm和Batch Norm的区别和使用场景？
8. 多头注意力机制的作用是什么？

## 4. 学习计划
- 每周学习20道算法题
- 每天复习5个LLM核心概念
- 每月进行一次模拟面试
- 重点掌握：Transformer、Attention、预训练模型

## 5. 面试技巧
- 思路清晰，有条理地回答问题
- 先思考再回答，不要急于给出答案
- 对于不确定的问题，可以先说明思路再尝试解答
- 展现自己的学习能力和思考方式
- 对于核心概念（如Transformer），准备好画图讲解

## 6. 参考资料
- 飞书文档：[xfan_面试强化](https://my.feishu.cn/docs/doccnqO8g5vIdVAgJ0PXbYNsJWe)
- 《百面自然语言处理》
- 《深度学习》花书
- Transformer原始论文：Attention Is All You Need