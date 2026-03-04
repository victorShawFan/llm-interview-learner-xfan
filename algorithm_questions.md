# 算法面试高频题详解

## 📌 动态规划

### 1. 最大子数组和（LeetCode 53）
**题目**：给定一个整数数组，找到一个具有最大和的连续子数组。

```python
def maxSubArray(nums):
    """
    动态规划解法
    dp[i] 表示以 nums[i] 结尾的最大子数组和
    """
    if not nums:
        return 0
    
    max_sum = nums[0]
    current_sum = nums[0]
    
    for i in range(1, len(nums)):
        current_sum = max(nums[i], current_sum + nums[i])
        max_sum = max(max_sum, current_sum)
    
    return max_sum

# 时间复杂度: O(n)
# 空间复杂度: O(1)
```

**面试技巧**：
- 说明Kadane算法思想
- 可以先说O(n²)暴力解法，再优化到O(n)

### 2. 打家劫舍（LeetCode 198）
**题目**：房屋相邻不能同时偷，求最大金额。

```python
def rob(nums):
    """
    dp[i] = max(dp[i-1], dp[i-2] + nums[i])
    """
    if not nums:
        return 0
    if len(nums) == 1:
        return nums[0]
    
    prev2 = nums[0]
    prev1 = max(nums[0], nums[1])
    
    for i in range(2, len(nums)):
        current = max(prev1, prev2 + nums[i])
        prev2 = prev1
        prev1 = current
    
    return prev1

# 时间复杂度: O(n)
# 空间复杂度: O(1)
```

### 3. 最长递增子序列（LeetCode 300）
**题目**：找到最长严格递增子序列的长度。

```python
def lengthOfLIS(nums):
    """
    dp[i] 表示以 nums[i] 结尾的最长递增子序列长度
    """
    if not nums:
        return 0
    
    dp = [1] * len(nums)
    
    for i in range(1, len(nums)):
        for j in range(i):
            if nums[i] > nums[j]:
                dp[i] = max(dp[i], dp[j] + 1)
    
    return max(dp)

# 时间复杂度: O(n²)
# 空间复杂度: O(n)

# 优化版：二分查找 O(nlogn)
def lengthOfLIS_optimized(nums):
    import bisect
    tails = []
    
    for num in nums:
        pos = bisect.bisect_left(tails, num)
        if pos == len(tails):
            tails.append(num)
        else:
            tails[pos] = num
    
    return len(tails)
```

---

## 📌 双指针

### 4. 三数之和（LeetCode 15）
**题目**：找出所有和为0的三元组。

```python
def threeSum(nums):
    """
    排序 + 双指针
    """
    nums.sort()
    result = []
    
    for i in range(len(nums) - 2):
        # 跳过重复元素
        if i > 0 and nums[i] == nums[i-1]:
            continue
        
        left, right = i + 1, len(nums) - 1
        
        while left < right:
            total = nums[i] + nums[left] + nums[right]
            
            if total < 0:
                left += 1
            elif total > 0:
                right -= 1
            else:
                result.append([nums[i], nums[left], nums[right]])
                
                # 跳过重复元素
                while left < right and nums[left] == nums[left+1]:
                    left += 1
                while left < right and nums[right] == nums[right-1]:
                    right -= 1
                
                left += 1
                right -= 1
    
    return result

# 时间复杂度: O(n²)
# 空间复杂度: O(1)
```

### 5. 接雨水（LeetCode 42）
**题目**：计算能接多少雨水。

```python
def trap(height):
    """
    双指针法
    """
    if not height:
        return 0
    
    left, right = 0, len(height) - 1
    left_max, right_max = 0, 0
    water = 0
    
    while left < right:
        if height[left] < height[right]:
            if height[left] >= left_max:
                left_max = height[left]
            else:
                water += left_max - height[left]
            left += 1
        else:
            if height[right] >= right_max:
                right_max = height[right]
            else:
                water += right_max - height[right]
            right -= 1
    
    return water

# 时间复杂度: O(n)
# 空间复杂度: O(1)
```

---

## 📌 二叉树

### 6. 二叉树最大路径和（LeetCode 124）
**题目**：路径可以从任意节点开始和结束。

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def maxPathSum(root):
    """
    递归 + 全局变量
    """
    max_sum = float('-inf')
    
    def max_gain(node):
        nonlocal max_sum
        
        if not node:
            return 0
        
        # 递归计算左右子节点的最大贡献值
        # 只有在最大贡献值大于0时才计入
        left_gain = max(max_gain(node.left), 0)
        right_gain = max(max_gain(node.right), 0)
        
        # 当前节点的最大路径和
        price_newpath = node.val + left_gain + right_gain
        
        # 更新答案
        max_sum = max(max_sum, price_newpath)
        
        # 返回节点的最大贡献值
        return node.val + max(left_gain, right_gain)
    
    max_gain(root)
    return max_sum

# 时间复杂度: O(n)
# 空间复杂度: O(h) h为树的高度
```

### 7. 二叉树的层序遍历（LeetCode 102）
```python
from collections import deque

def levelOrder(root):
    """
    BFS队列
    """
    if not root:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        level_size = len(queue)
        current_level = []
        
        for _ in range(level_size):
            node = queue.popleft()
            current_level.append(node.val)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        result.append(current_level)
    
    return result

# 时间复杂度: O(n)
# 空间复杂度: O(n)
```

---

## 📌 图算法

### 8. 岛屿数量（LeetCode 200）
**题目**：计算二维网格中岛屿的数量。

```python
def numIslands(grid):
    """
    DFS
    """
    if not grid:
        return 0
    
    def dfs(i, j):
        if (i < 0 or i >= len(grid) or 
            j < 0 or j >= len(grid[0]) or 
            grid[i][j] == '0'):
            return
        
        grid[i][j] = '0'  # 标记为已访问
        
        # 访问四个方向
        dfs(i+1, j)
        dfs(i-1, j)
        dfs(i, j+1)
        dfs(i, j-1)
    
    count = 0
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == '1':
                dfs(i, j)
                count += 1
    
    return count

# 时间复杂度: O(m*n)
# 空间复杂度: O(m*n)
```

---

## 📌 NLP/CV特有算法

### 9. NMS（非极大值抑制）
**应用**：目标检测中去除重复框。

```python
def nms(boxes, scores, iou_threshold=0.5):
    """
    boxes: [[x1, y1, x2, y2], ...]
    scores: [score1, score2, ...]
    """
    # 按置信度排序
    indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    
    keep = []
    
    while indices:
        current = indices[0]
        keep.append(current)
        indices = indices[1:]
        
        # 计算当前框与剩余框的IoU
        remaining = []
        for idx in indices:
            iou = calculate_iou(boxes[current], boxes[idx])
            if iou < iou_threshold:
                remaining.append(idx)
        
        indices = remaining
    
    return keep

def calculate_iou(box1, box2):
    """计算两个框的IoU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0
```

**面试要点**：
- 解释NMS的作用和原理
- IoU计算公式
- Soft-NMS的改进

### 10. 最长公共子序列（LCS）
**应用**：文本相似度、DNA序列比对。

```python
def longestCommonSubsequence(text1, text2):
    """
    dp[i][j] 表示 text1[0:i] 和 text2[0:j] 的LCS长度
    """
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[m][n]

# 时间复杂度: O(m*n)
# 空间复杂度: O(m*n)
```

---

## 🎯 刷题策略

### 按难度分配
- **简单题**：30%（基础语法、数组操作）
- **中等题**：60%（动态规划、DFS/BFS、双指针）
- **困难题**：10%（复杂DP、图算法）

### 按类型分配
1. 动态规划：20道
2. 双指针/滑动窗口：15道
3. 二叉树/图：15道
4. 链表：10道
5. 排序/查找：10道
6. 字符串：10道
7. NLP/CV特有：10道
8. 其他：10道

### 每日计划
- 第1周：每天3道简单题
- 第2-3周：每天2道中等题
- 第4-5周：每天1道困难题 + 1道中等题
- 第6周：模拟面试，复习错题

---

## 📚 参考资源
- LeetCode中国：https://leetcode.cn/
- 《剑指Offer》
- 《程序员代码面试指南》
- AlgoExpert
