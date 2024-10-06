from typing import List

class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        num_table = {}
        for i in range(len(nums)):
            if target - nums[i] not in num_table:
                num_table[nums[i]] = i
            else:
                return [i, num_table[target - nums[i]]]
        return False

solution = Solution()

# 示例 1
nums1 = [2, 7, 11, 15]
target1 = 9
result1 = solution.twoSum(nums1, target1)
print(f"输入：nums = {nums1}, target = {target1}\n输出：{result1}\n")

# 示例 2
nums2 = [3, 2, 4]
target2 = 6
result2 = solution.twoSum(nums2, target2)
print(f"输入：nums = {nums2}, target = {target2}\n输出：{result2}\n")

# 示例 3
nums3 = [3, 3]
target3 = 6
result3 = solution.twoSum(nums3, target3)
print(f"输入：nums = {nums3}, target = {target3}\n输出：{result3}\n")
