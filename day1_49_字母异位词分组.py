from typing import List
from collections import Counter, defaultdict

class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        res = defaultdict(list)
        for str in strs:
            str_table = tuple(sorted(Counter(str).items()))  # 将字母出现次数排序并转换为元组，作为键
            res[str_table].append(str)
        return list(res.values())

solution = Solution()

# 示例 1
strs1 = ["eat", "tea", "tan", "ate", "nat", "bat"]
result1 = solution.groupAnagrams(strs1)
print(f"输入: strs = {strs1}\n输出: {result1}\n")

# 示例 2
strs2 = [""]
result2 = solution.groupAnagrams(strs2)
print(f"输入: strs = {strs2}\n输出: {result2}\n")

# 示例 3
strs3 = ["a"]
result3 = solution.groupAnagrams(strs3)
print(f"输入: strs = {strs3}\n输出: {result3}\n")
