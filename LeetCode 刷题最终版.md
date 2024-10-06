# LeetCode 刷题最终版

2024.9.19 - 

## 双指针

2024.9.19 - 

### 1.有序数组的 Two sum

2024.9.19

```python
class Solution:
    def twoSum(self, numbers, target):
        left, right = 0, len(numbers) - 1
        # cur_sum = 0
        while left < right:
            cur_sum = numbers[left] + numbers[right]
            if cur_sum == target:
                return [left + 1, right + 1]
            elif cur_sum > target:
                right -= 1
            else:
                left += 1
        return []
```

### 2.两数平方和

2024.9.19

```python
class Solution:
    def judgeSquareSum(self, c: int) -> bool:
        left, right = 0, int(sqrt(c)) ## 向下取整
        while left <= right:
            cur_sum = left**2 + right**2
            if cur_sum == c:
                return True
            elif cur_sum > c:
                right -= 1
            else:
                left += 1 
        return False
```

### 3.反转字符串中的元音字符

2024.9.19

```python
# 输入：s = "IceCreAm"
# 输出："AceCreIm"
class Solution:
    def reverseVowels(self, s: str) -> str:
        s = list(s)
        target_char = ['a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U']
        left, right = 0, len(s) - 1
        while left < right:
            # 避免内部交叉
            while left < right and s[right] not in target_char:
                right -= 1
            while left < right and s[left] not in target_char:
                left += 1
            if s[left] in target_char and s[right] in target_char:
                s[left], s[right] = s[right], s[left]
                left += 1
                right -= 1
        return ''.join(s)                   
```

## 数组

### 283. 移动零

要在原数组上修改

```python
class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        temp = 0
        for i in range(len(nums)):
            if nums[i] != 0:
                nums[temp] = nums[i]
                temp += 1
                
        for i in range(temp, len(nums)):
            nums[i] = 0
        return nums
```

### 566. 重塑矩阵

```python
class Solution:
    def matrixReshape(self, mat: List[List[int]], r: int, c: int) -> List[List[int]]:
        m = len(mat) # hang
        n = len(mat[0]) # lie
        if m*n != r*c:
            return mat
        temp = 0
        num_matrix = []
        for i in range(m):
            for j in range(n):
                num_matrix.append(mat[i][j])
        res = [[0] * c for _ in range(r)] 
        for i in range(r):
            for j in range(c):
                res[i][j] = num_matrix[temp]
                temp += 1
        return res

```

### 485. 最大连续1的个数

想复杂了：[1,1,0,1,1,1] 出现0以后在此之前不可能出现更长的连续1了

```python
class Solution:
    def findMaxConsecutiveOnes(self, nums: List[int]) -> int:
        max_len = 0
        cur_len = 0
        for num in nums:
            if num == 1:
                cur_len += 1
                max_len = max(cur_len, max_len)
            else:
                cur_len = 0
        return max_len


```

### 240. 搜索二维矩阵I

```python
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        if not matrix or not matrix[0]:
            return False
        
        m = len(matrix)    # 行数
        n = len(matrix[0]) # 列数
        row, column = 0, n - 1  # 从右上角开始搜索        
        while row < m and column >= 0: 
            if matrix[row][column] == target:
                return True
            elif matrix[row][column] < target:
                row += 1  # 向下移动
            else:
                column -= 1  # 向左移动
                
        return False

```

### 697. 数组的度

有点难 2024.10.3

```python
class Solution:
    def findShortestSubArray(self, nums: List[int]) -> int:
        from collections import Counter
        num_table = Counter(nums)
        degree = max(num_table.values())
        start_index = {}
        last_index = {}
        for i,num in enumerate(nums):
            if num_table[num] == degree:
                if num not in start_index:
                    start_index[num] = i
                last_index[num] = i
        min_len = float('inf')

        for num in start_index:
            length = last_index[num] - start_index[num] + 1
            min_len = min(min_len,length)
        return min_len

```

### 645. 错误的集合

```python
class Solution:
    def findErrorNums(self, nums: List[int]) -> List[int]:
        n = len(nums)   
        # 统计每个数字的出现次数
        for num in nums:
            num_dict[num] = num_dict.get(num, 0) + 1
            
        res = []
        duplicate = -1
        missing = -1
        
        # 找出重复的数字和缺失的数字
        for i in range(1, n + 1):
            if i in num_dict:
                if num_dict[i] == 2:
                    duplicate = i  # 找到重复的数字
            else:
                missing = i  # 找到缺失的数字
        
        return [duplicate, missing]
```

### 766. 托普利茨矩阵

```python
class Solution:
    def isToeplitzMatrix(self, matrix) -> bool:
        n = len(matrix[0]) #lie
        m = len(matrix) # hang
        print(m, n)
        for i in range(m):
            for j in range(n):
                if i + 1 < m and j + 1 < n:
                    while matrix[i][j] != matrix[i+1][j+1]:
                        return False
        return True

```

### 769. 最多能完成排序的块

```python
class Solution:
    def maxChunksToSorted(self, arr) -> int:
        length = len(arr)
        cur_max = 0
        chuck_num = 0
        for i in range(length):
            cur_max = max(arr[i], cur_max)
            print(cur_max, i)
            if cur_max == i :
                chuck_num += 1

        return chuck_num
```

### 378. 有序矩阵中第K小的元素

有点难

```python
class Solution:
    def kthSmallest(self, matrix: List[List[int]], k: int) -> int:
        def count_num(mid):
            # 计算矩阵中小于等于 mid 的元素个数
            count = 0
            row, col = n - 1, 0
            
            # 从矩阵的左下角开始
            while row >= 0 and col < n:
                if matrix[row][col] <= mid:
                    count += (row + 1)
                    col += 1
                else:
                    row -= 1
            return count

        n = len(matrix)
        left, right = matrix[0][0], matrix[n-1][n-1]
        while left <= right:
            mid = left + (right - left) // 2
            if count_num(mid) < k:
                left = mid + 1
            else:
                right = mid - 1
        return left
```

## 二分查找

### 33. 搜索旋转排序数组

```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        # 第一步：找到旋转点
        def find_rotation_point(nums):
            left, right = 0, len(nums) - 1
            while left < right:
                mid = left + (right - left) // 2
                # 判断 mid 是否是旋转点
                if nums[mid] > nums[right]:
                    left = mid + 1  # 旋转点在右半部分
                else:
                    right = mid # 旋转点在左半部分
            return left  # 返回旋转点的索引

        rotation_index = find_rotation_point(nums)
        
        # 第二步：根据目标值进行二分查找
        left, right = 0, len(nums) - 1
        
        # 确定目标值在左半部分还是右半部分
        if target >= nums[rotation_index] and target <= nums[right]:
            left = rotation_index
        else:
            right = rotation_index - 1
        
        # 进行二分查找
        while left <= right:
            mid = left + (right - left) // 2
            
            if nums[mid] == target:
                return mid
            elif nums[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        
        return -1  # 如果未找到

```

## 哈希表

### 1. 两数之和

不知道这种题为什么不能秒过。像个废物。  2024.10.6

```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        num_table = {}
        for i in range(len(nums)):
            if target - nums[i] not in num_table:
                num_table[nums[i]] = i
            else:
                return [i, num_table[target - nums[i]]]
        return False
```

### 49. 字母异位词分组

有点难，是好题

本质上这个代码是将通过 `Counter` 和 `sorted().items()` 得到的字母及其频率的排序结果作为哈希表的键。

```python
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        from collections import Counter, defaultdict
        str_dict = defaultdict(list)
        for str in strs:
            count = tuple(sorted(Counter(str).items()))
            str_dict[count].append(str)
        return list(str_dict.values())

```

### 128. 最长连续序列

```python
class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        set_nums = set(nums)
        cur_num = 0
        cur_count = 0
        max_count = 0
        for num in set_nums:
            if num - 1 not in set_nums:
                cur_num = num
                cur_count = 1
            while cur_num + 1 in set_nums:
                cur_count += 1
                cur_num += 1
            max_count = max(cur_count, max_count)
        return max_count

```

