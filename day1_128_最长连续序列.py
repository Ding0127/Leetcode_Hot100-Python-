from typing import List


class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        num_set = set(nums)
        max_length = 0

        for num in num_set:
            # 只有当前数字是序列的起点时，才开始统计长度
            if num - 1 not in num_set:
                current_num = num
                current_count = 1

                # 找出当前序列的长度
                while current_num + 1 in num_set:
                    current_num += 1
                    current_count += 1

                # 更新最长序列长度
                max_length = max(max_length, current_count)

        return max_length


solution = Solution()

# 示例 1
nums1 = [100, 4, 200, 1, 3, 2]
result1 = solution.longestConsecutive(nums1)
print(f"输入: nums = {nums1}\n输出: {result1}\n")

# 示例 2
nums2 = [0, 3, 7, 2, 5, 8, 4, 6, 0, 1]
result2 = solution.longestConsecutive(nums2)
print(f"输入: nums = {nums2}\n输出: {result2}\n")
