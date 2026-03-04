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

## 2. LLM常见面试问题
1. LLM的训练过程是怎样的？
2. 如何解决过拟合问题？
3. 注意力机制的原理是什么？
4. 你对大模型的伦理问题有什么看法？

## 3. 学习计划
- 每周学习20道算法题
- 每天复习5个LLM核心概念
- 每月进行一次模拟面试

## 4. 面试技巧
- 思路清晰，有条理地回答问题
- 先思考再回答，不要急于给出答案
- 对于不确定的问题，可以先说明思路再尝试解答
- 展现自己的学习能力和思考方式