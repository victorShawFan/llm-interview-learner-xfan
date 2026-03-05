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

## 📌 滑动窗口

### 11. 滑动窗口最大值（LeetCode 239）
**题目**：给定数组和窗口大小k，返回每个窗口中的最大值。

```python
from collections import deque

def maxSlidingWindow(nums, k):
    """
    单调递减队列
    """
    if not nums:
        return []
    
    result = []
    dq = deque()  # 存储索引
    
    for i in range(len(nums)):
        # 移除窗口外的元素
        if dq and dq[0] < i - k + 1:
            dq.popleft()
        
        # 维护单调递减：移除比当前元素小的
        while dq and nums[dq[-1]] < nums[i]:
            dq.pop()
        
        dq.append(i)
        
        # 窗口形成后开始记录结果
        if i >= k - 1:
            result.append(nums[dq[0]])
    
    return result

# 示例
nums = [1,3,-1,-3,5,3,6,7]
k = 3
print(maxSlidingWindow(nums, k))
# 输出: [3,3,5,5,6,7]

# 时间复杂度: O(n)
# 空间复杂度: O(k)
```

**面试技巧**：
- 说明为什么用单调队列
- 时间复杂度从O(n*k)优化到O(n)

### 12. 无重复字符的最长子串（LeetCode 3）
**题目**：找出不含有重复字符的最长子串的长度。

```python
def lengthOfLongestSubstring(s):
    """
    滑动窗口 + 哈希表
    """
    if not s:
        return 0
    
    char_index = {}  # 记录字符最后出现的位置
    max_len = 0
    left = 0
    
    for right in range(len(s)):
        char = s[right]
        
        # 如果字符已存在且在窗口内，移动左边界
        if char in char_index and char_index[char] >= left:
            left = char_index[char] + 1
        
        char_index[char] = right
        max_len = max(max_len, right - left + 1)
    
    return max_len

# 示例
print(lengthOfLongestSubstring("abcabcbb"))  # 3 ("abc")
print(lengthOfLongestSubstring("bbbbb"))     # 1 ("b")

# 时间复杂度: O(n)
# 空间复杂度: O(min(m, n)) m为字符集大小
```

### 13. 最小覆盖子串（LeetCode 76）
**题目**：找出s中涵盖t所有字符的最小子串。

```python
from collections import Counter

def minWindow(s, t):
    """
    滑动窗口 + 计数器
    """
    if not s or not t:
        return ""
    
    # 统计t中字符频率
    need = Counter(t)
    window = {}
    
    left = 0
    valid = 0  # 已满足条件的字符数
    start = 0
    min_len = float('inf')
    
    for right in range(len(s)):
        char = s[right]
        
        # 加入窗口
        if char in need:
            window[char] = window.get(char, 0) + 1
            if window[char] == need[char]:
                valid += 1
        
        # 尝试收缩窗口
        while valid == len(need):
            # 更新最小窗口
            if right - left + 1 < min_len:
                start = left
                min_len = right - left + 1
            
            # 移除左边界字符
            remove_char = s[left]
            if remove_char in need:
                if window[remove_char] == need[remove_char]:
                    valid -= 1
                window[remove_char] -= 1
            
            left += 1
    
    return s[start:start+min_len] if min_len != float('inf') else ""

# 示例
print(minWindow("ADOBECODEBANC", "ABC"))  # "BANC"

# 时间复杂度: O(|s| + |t|)
# 空间复杂度: O(|s| + |t|)
```

### 14. 找到字符串中所有字母异位词（LeetCode 438）
**题目**：找到s中所有p的异位词的起始索引。

```python
from collections import Counter

def findAnagrams(s, p):
    """
    滑动窗口 + 计数器
    """
    if len(s) < len(p):
        return []
    
    need = Counter(p)
    window = {}
    
    result = []
    left = 0
    valid = 0
    
    for right in range(len(s)):
        char = s[right]
        
        if char in need:
            window[char] = window.get(char, 0) + 1
            if window[char] == need[char]:
                valid += 1
        
        # 窗口大小等于p的长度
        if right - left + 1 == len(p):
            if valid == len(need):
                result.append(left)
            
            # 移除左边界
            remove_char = s[left]
            if remove_char in need:
                if window[remove_char] == need[remove_char]:
                    valid -= 1
                window[remove_char] -= 1
            
            left += 1
    
    return result

# 示例
print(findAnagrams("cbaebabacd", "abc"))  # [0, 6]
```

### 15. 滑动窗口中位数（LeetCode 480）
**题目**：计算滑动窗口的中位数。

```python
import heapq

def medianSlidingWindow(nums, k):
    """
    双堆法：大顶堆存小的一半，小顶堆存大的一半
    """
    # Python只有小顶堆，用负数模拟大顶堆
    max_heap = []  # 存较小的一半（负数）
    min_heap = []  # 存较大的一半
    
    # 延迟删除字典
    to_remove = {}
    
    def prune(heap, is_max_heap):
        """清理堆顶已删除的元素"""
        while heap:
            num = -heap[0] if is_max_heap else heap[0]
            if (is_max_heap and -heap[0] in to_remove and to_remove[-heap[0]] > 0) or \
               (not is_max_heap and heap[0] in to_remove and to_remove[heap[0]] > 0):
                actual_num = -heapq.heappop(heap) if is_max_heap else heapq.heappop(heap)
                to_remove[actual_num] -= 1
            else:
                break
    
    def get_median():
        if k % 2 == 1:
            return float(-max_heap[0])
        else:
            return (-max_heap[0] + min_heap[0]) / 2.0
    
    # 初始化第一个窗口
    for i in range(k):
        heapq.heappush(max_heap, -nums[i])
    
    # 平衡堆：将max_heap的一半移到min_heap
    for _ in range(k // 2):
        heapq.heappush(min_heap, -heapq.heappop(max_heap))
    
    result = [get_median()]
    
    for i in range(k, len(nums)):
        out_num = nums[i - k]
        in_num = nums[i]
        balance = 0
        
        # 删除离开窗口的数
        if out_num <= -max_heap[0]:
            balance -= 1
        else:
            balance += 1
        
        to_remove[out_num] = to_remove.get(out_num, 0) + 1
        
        # 插入新数
        if in_num <= -max_heap[0]:
            heapq.heappush(max_heap, -in_num)
            balance += 1
        else:
            heapq.heappush(min_heap, in_num)
            balance -= 1
        
        # 平衡堆
        if balance < 0:
            heapq.heappush(max_heap, -heapq.heappop(min_heap))
        elif balance > 0:
            heapq.heappush(min_heap, -heapq.heappop(max_heap))
        
        prune(max_heap, True)
        prune(min_heap, False)
        
        result.append(get_median())
    
    return result

# 时间复杂度: O(n * logk)
# 空间复杂度: O(k)
```

---

## 📌 链表操作

### 16. 反转链表（LeetCode 206）
**题目**：反转一个单链表。

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def reverseList(head):
    """
    迭代法
    """
    prev = None
    current = head
    
    while current:
        next_temp = current.next
        current.next = prev
        prev = current
        current = next_temp
    
    return prev

# 递归法
def reverseList_recursive(head):
    if not head or not head.next:
        return head
    
    new_head = reverseList_recursive(head.next)
    head.next.next = head
    head.next = None
    
    return new_head

# 时间复杂度: O(n)
# 空间复杂度: 迭代O(1), 递归O(n)
```

### 17. 合并两个有序链表（LeetCode 21）
**题目**：将两个升序链表合并为一个新的升序链表。

```python
def mergeTwoLists(l1, l2):
    """
    迭代法
    """
    dummy = ListNode(0)
    current = dummy
    
    while l1 and l2:
        if l1.val <= l2.val:
            current.next = l1
            l1 = l1.next
        else:
            current.next = l2
            l2 = l2.next
        current = current.next
    
    current.next = l1 if l1 else l2
    
    return dummy.next

# 递归法
def mergeTwoLists_recursive(l1, l2):
    if not l1:
        return l2
    if not l2:
        return l1
    
    if l1.val <= l2.val:
        l1.next = mergeTwoLists_recursive(l1.next, l2)
        return l1
    else:
        l2.next = mergeTwoLists_recursive(l1, l2.next)
        return l2

# 时间复杂度: O(n + m)
# 空间复杂度: 迭代O(1), 递归O(n + m)
```

### 18. 环形链表（LeetCode 141）
**题目**：判断链表是否有环。

```python
def hasCycle(head):
    """
    快慢指针
    """
    if not head or not head.next:
        return False
    
    slow = head
    fast = head.next
    
    while slow != fast:
        if not fast or not fast.next:
            return False
        slow = slow.next
        fast = fast.next.next
    
    return True

# 时间复杂度: O(n)
# 空间复杂度: O(1)
```

### 19. 环形链表II（LeetCode 142）
**题目**：找到环的入口节点。

```python
def detectCycle(head):
    """
    快慢指针 + 数学推导
    """
    if not head or not head.next:
        return None
    
    # 第一次相遇
    slow = head
    fast = head
    
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        
        if slow == fast:
            # 从相遇点和头节点同时出发，第二次相遇即为环入口
            slow = head
            while slow != fast:
                slow = slow.next
                fast = fast.next
            return slow
    
    return None

# 数学原理：
# 设环前长度为a，环长度为b
# 第一次相遇时：slow走了s步，fast走了2s步
# 2s = s + nb => s = nb
# 环入口：从head走a步，从相遇点走a步也到达环入口

# 时间复杂度: O(n)
# 空间复杂度: O(1)
```

### 20. 删除链表的倒数第N个节点（LeetCode 19）
**题目**：删除链表的倒数第n个节点。

```python
def removeNthFromEnd(head, n):
    """
    双指针
    """
    dummy = ListNode(0)
    dummy.next = head
    
    fast = dummy
    slow = dummy
    
    # fast先走n步
    for _ in range(n):
        fast = fast.next
    
    # 同时移动直到fast到最后
    while fast.next:
        slow = slow.next
        fast = fast.next
    
    # 删除节点
    slow.next = slow.next.next
    
    return dummy.next

# 时间复杂度: O(n)
# 空间复杂度: O(1)
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
