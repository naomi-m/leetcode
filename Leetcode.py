from typing import List
import ListNode
import TreeNode
    
class Leetcode:

    def strReverse(new_input_string, new_input_string_reverse):
        for ch in new_input_string:
            new_input_string_reverse = ch + new_input_string_reverse

        print("1.", new_input_string)
        print("2.", new_input_string_reverse)

    # strReverse("abcdef","")


    def twoSum(nums, target):
        """
        param nums List[int]
        param target int
        return List[int]
        """
        # first loop pointer for first num to compare
        # second loop pointer for second num to compare
        answer = []  # empty list if not found
        listLen = len(nums)
        for i in range(listLen-1):
            for j in range(i+1, listLen):
                if nums[i] + nums[j] == target:
                    answer = [i,j]  # update index nums in answer
                    break  # found answer so stop searching
        # return answer
        print("twoSum: ",answer)

    # twoSum([2,7,11,15], 9)


    def climbStairs(n):
        # You are climbing a staircase. It takes n steps to reach the top.
        # Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?
        # Example 1:
            # Input: n = 2
            # Output: 2
            # Explanation: There are two ways to climb to the top.
            # 1. 1 step + 1 step
            # 2. 2 steps
        # Example 2:
            # Input: n = 3
            # Output: 3
            # Explanation: There are three ways to climb to the top.
            # 1. 1 step + 1 step + 1 step
            # 2. 1 step + 2 steps
            # 3. 2 steps + 1 step
        """
        :type n: int
        :rtype: int
        """
        # recursion with memoization
        memo = {}  # create memo list
        memo[1] = 1  # insert known steps
        memo[2] = 2  # insert known steps
        
        # memoization table stores already-calculated
        def climb(n, memo):
            # base case 1 option when 1 step or 2 options when 2 steps
            if n == 1:
                return 1
            if n == 2:
                return 2
            if n not in memo:
                memo[n] = climb(n-1, memo) + climb(n-2, memo)
            return memo[n]
        
        return climb(n, memo)
    

    def reverseList(self, head):
        # print("climbing stairs: ", climbStairs(3))

        # Definition for singly-linked list.
        # class ListNode(object):
        #     def __init__(self, val=0, next=None):
        #         self.val = val
        #         self.next = next
        """
        :type head: ListNode
        :rtype: ListNode
        """
        # iterative replacement of each node; continue until next node pointer is null bc Tail pointer will be None so loop will end
        # store next node in temp so don't lose next node
        # adjust next node pointer of current node (if current head, becomes null else previous node)
        prev = None
        while head:
            current = head
            head = head.next
            current.next = prev
            prev = current
        return prev


    def insertionSortList(head):
        # Definition for singly-linked list.
        # class ListNode(object):
        #     def __init__(self, val=0, next=None):
        #         self.val = val
        #         self.next = next    
        """
        :type head: ListNode
        :rtype: ListNode
        """
        # Insertion sort iterates, consuming one input element each repetition and growing a sorted output list.
        # At each iteration, insertion sort removes one element from the input data, finds the location it belongs within the sorted list and inserts it there.
        # It repeats until no input elements remain.

        # cases: insert at front, middle, end, no need to insert

        dummyHead = ListNode(0)  # create dummy in case need to insert at head
        dummyHead.next = head
        sortEnd = head

        # iterate through LL if node.next is not None
        while sortEnd.next:
            # if in order iterate forward else re-order
            if sortEnd < sortEnd.next:
                sortEnd = sortEnd.next  # update ptr
            else:
                while sortEnd > sortEnd.next:
                # compare from start each node against curr
                # when correct place, insert curr into place, update pointers so don't lose linked list
                    beforeInsert = dummyHead
                    curr = sortEnd.next
                    if beforeInsert < curr and curr < beforeInsert.next:
                        sortEnd.next = curr.next  # end points to node after curr
                        curr.next = beforeInsert.next  # curr points to node that should be after insert
                        beforeInsert.next = curr  # node before insert points to insert
                        if beforeInsert == dummyHead:
                            head = beforeInsert.next
                            dummyHead.next = head
                    else:
                        beforeInsert = beforeInsert.next  # iterate forward to check correct place
        return dummyHead.next
    

    def addBinary(a, b):
        """
        :type a: str
        :type b: str
        :rtype: str
        """
        # binary addition: starts from right-aligned, if two 1's then carryover 1 to next column
        result = ""
        lenA = len(a)
        lenB = len(b)
        diff = abs(lenA - lenB)

        # pad small string with 0's at front to equal len of big string
        pad = "0" * (diff + 1)  # +1 accounts for extra LHS carryover padding
        if lenA < lenB:
            a = pad + a
            b = "0" + b
        else:
            b = pad + b
            a = "0" + a
        
        # loop [now equal len] strings backwards, if a+b=2 then carryover to next loop else keep answer 0 or 1
        maxLen = max(lenA, lenB) + 1
        carryover = 0
        for i in range(maxLen - 1, -1, -1):
            val = int(a[i]) + int(b[i]) + carryover
            carryover = 0  # reset
            if val == 2:
                carryover = 1
                val = 0
            if val == 3:
                carryover = 1
                val = 1
            result = str(val) + result
        if result[0] == "0": result = result[1:]  # strip leading 0 if needed
        # return result
        print(result)

    # addBinary("1010","1011")


    def searchInsert(nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        # binary search, recursion solution
        # l is left index, r is right index
        n = len(nums)
        l = 0
        r = n - 1
        # loop until list has one val to search
        while l < r:
            m = int( (l + r) / 2 )  # middle
            if nums[m] == target:
                return m
            elif nums[m] < target:
                l = m + 1
            else:
                r = m
        # result is to the right if >, else if <= then result is current index
        return l + 1 if nums[l] < target else l

    # print( searchInsert([1,3,5,6], 2) )


    def isHappy(n):
        """
        :type n: int
        :rtype: bool
        """
        # Starting with any positive integer, replace the number by the sum of the squares of its digits.
        # Repeat the process until the number equals 1 (where it will stay), or it loops endlessly in a cycle which does not include 1.
        # Those numbers for which this process ends in 1 are happy.
        # Return true if n is a happy number, and false if not.

        visited = set()  # int set ensures no duplicates
        # repeat while not seen before, if seen means infinte loop
        while n not in visited:
            visited.add(n)  # add to set
            nStr = str(n)  # convert int to str to iterate
            n = 0  # reset before replace
            for x in nStr:  # replace
                n += int(x) ** 2
            if n == 1:  # check if happy to stop, else continue
                return True
        return False
    

    def addTwoNumbers(l1, l2):
        # Definition for singly-linked list.
        # class ListNode(object):
        #     def __init__(self, val=0, next=None):
        #         self.val = val
        #         self.next = next
        """
        :type l1: Optional[ListNode]
        :type l2: Optional[ListNode]
        :rtype: Optional[ListNode]
        """
        # input l1, l2 into str reverse
        # cast str to int
        # do math
        # cast int to str
        # iterate reverse str to create result
        l1Temp = ""
        l2Temp = ""
        while l1 != None:
            l1Temp = str(l1.val) + l1Temp  # explicit cast, concat to front to reverse
            l1 = l1.next
        while l2 != None:
            l2Temp = str(l2.val) + l2Temp
            l2 = l2.next
        l1Int = int(l1Temp)
        l2Int = int(l2Temp)
        sumWrongOrder = str(l1Int + l2Int)
        sum = ""
        for x in sumWrongOrder:
            sum = x + sum
        n = len(sum)
        if n == 1:  # for loop only works min len 2
            return ListNode(int(sum[0]), None)

        result = ListNode(-1, None)  # placeholder
        temp = ListNode(-1, None)  # placeholder
        for i in range(1, n):
            new = ListNode(int(sum[i]), None)
            if i == 1:  # head only
                result = ListNode(int(sum[i-1]), new)
            temp.next = new
            temp = new
        return result
    

    def sortArray(nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        # mergesort: divide in halves until single element, sort, merge

        def merge(nums, l, m, r):
            """
            Params:
                nums List[int]
                l int left-most index
                m int middle index
                r int right-most index
            Returns:
                n/a
            """
            # get size of each subarray
            n1 = m - l + 1
            n2 = r - m

            # create temp subarr copying over from nums; list comprehension 
            arr1 = [nums[i] for i in range(l, m + 1) if True]
            arr2 = [nums[i] for i in range(m + 1, r + 1) if True]
            # arr1 = [nums[l + i] for i in range(n1) if True]
            # arr2 = [nums[m + 1 + i] for i in range(n2) if True]

            # merge temp subarr
            i = 0  # initial index subarr 1
            j = 0  # initial index subarr 2
            k = l  # initial index merged subarr
            while i < n1 and j < n2:  # ensure index bounds
                # compare arr1 vs arr2 and choose lower val and overwrite og arr to become merged subarr
                # increment subarr index and merged arr index
                if arr1[i] <= arr2[j]:
                    nums[k] = arr1[i]
                    i += 1
                else:
                    nums[k] = arr2[j]
                    j += 1
                k += 1
            
            # if while loop ends bc one arr ends early copy remaining elements of other arr
            # uses similar logic as above
            while i < n1:
                nums[k] = arr1[i]
                i += 1
                k += 1
            while j < n2:
                nums[k] = arr2[j]
                j += 1
                k += 1

        def sort(nums, l, r):
            """
            Params:
                nums List[int]
                l int left-most index
                r int right-most index
            Returns:
                n/a
            """
            # ensure two elements since always l != r
            if l < r:
                # define m, sort l, sort r, merge l and r
                m = (l + r) // 2  # floor division (get whole num)
                sort(nums, l, m)
                sort(nums, m + 1, r)
                merge(nums, l, m, r)
        
        sort(nums, 0, len(nums) - 1)

        return nums  # sorts in-place
    
    # nums = [5,2,3,1]
    # print(nums)
    # sortArray(nums)
    # print(nums)


    def maxSubArray(nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # divide and conquer: l, r, or l-m-r combo
        min = -(pow(10,5))
        def maxSubArray(arr, l, r):
            if l > r: return min  # base case crossover, return min so not considered
            m = (l + r) // 2  # define middle
            lMax = 0
            currMax = 0
            rMax = 0
            # find max l combos, start from center step reverse
            for i in range(m - 1, l - 1, -1):
                lMax = max(lMax, currMax := currMax + arr[i])
            currMax = 0  # reset
            # find max r combos, start from center step normal
            for i in range(m + 1, r + 1):
                rMax = max(rMax, currMax := currMax + arr[i])
            # find max of l, r, l-m-r combo and return
            return max( maxSubArray(arr, l, m - 1), maxSubArray(arr, m + 1, r), lMax + arr[m] + rMax )
        
        return maxSubArray(nums, 0, len(nums) - 1)
    
    # nums = [-2,1,-3,4,-1,2,1,-5,4]
    # nums = [1]
    # print( maxSubArray(nums) )


    def constructMaximumBinaryTree(nums: List[int]) -> TreeNode:
        # Definition for a binary tree node.
        # class TreeNode:
        #     def __init__(self, val=0, left=None, right=None):
        #         self.val = val
        #         self.left = left
        #         self.right = right

        # recursion divide and conquer
        def maxBinT(arr, l, r):
            if l > r: return None  # base case

            # identify root (max) index; compare each element against known max, update as needed; O(n)
            iMax = 0  # represents 0th element
            for i in range(l, r + 1):
                if nums[i] > nums[iMax]: iMax = i
            
            # create l, r subarr trees
            # establish pointers
            left = maxBinT(arr, l, iMax - 1)
            right = maxBinT(arr, iMax + 1, r)
            return TreeNode(nums[iMax], left, right)

        # TODO need to debug looping
        return maxBinT(nums, 0, len(nums) - 1)

    # nums = [3,2,1,6,0,5]  # output is [6,3,5,null,2,0,null,null,1]
    # print( constructMaximumBinaryTree(nums) )


    def longestPalindrome(s):
        """
        :type s: str
        :rtype: str
        """
        # default max is single letter 0th index
        # check center-out and track max
        n = len(s)
        if n == 1: return s
        longest = s[0]

        def check(l, r):
            while l >= 0 and r < n and s[l] == s[r]:
                l -= 1
                r += 1
            return s[l+1:r]  # bc decrement l before break loop and slice is [incl:excl] so need to +1 to l

        for i in range(n-1):
            odd = check(i, i)
            even = check(i, i+1)

            if len(odd) > len(longest): longest = odd
            if len(even) > len(longest): longest = even

        return longest
    
    # s = "babad"
    # longestPalindrome(s)


    def rob(nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # memoization creation
        # call recursive, compare/choose max
        n = len(nums)
        memo = [-1] * n

        def evaluate(i):
            if i < 0: return 0
            if memo[i] >= 0: return memo[i]

            result = max( evaluate(i-2) + nums[i], evaluate(i-1) )
            memo[i] = result
            return result

        return evaluate(n-1)
    
    # nums = [1,2,3,1]
    # rob(nums)


    def spiralOrder(matrix):
        """
        :type matrix: List[List[int]]
        :rtype: List[int]
        """
        # reduce row, col boundaries and iterate
        row = 0
        col = 0
        m = len(matrix)  # row
        n = len(matrix[0])  # col
        rowStart = 0  # row start boundary
        colStart = 0  # col start boundary
        numElements = m * n
        result = []
        limit = len(matrix) // 2 if len(matrix) % 2 == 0 else (len(matrix) // 2) + 1

        for ct in range(limit):  # each box considers 2 rows
            while col < n:  # travel right
                result.append(matrix[row][col])
                col += 1
            if len(result) == numElements: break  # check after each row/col
            col -= 1  # so col is last index
            row += 1
            while row < m:  # travel down
                result.append(matrix[row][col])
                row += 1
            if len(result) == numElements: break

            row -= 1  # so row is last index
            col -= 1
            while col >= colStart:  # travel left
                result.append(matrix[row][col])
                col -= 1
            if len(result) == numElements: break

            col += 1  # so col is last index
            row -= 1
            while row > rowStart:  # travel up
                result.append(matrix[row][col])
                row -= 1
            if len(result) == numElements: break
            row += 1  # so row is last index
            col += 1

            m -= 1  # reduce boundaries
            n -= 1
            rowStart += 1
            colStart += 1
        return result
        
    # matrix = [[1,2,3],[4,5,6],[7,8,9]]
    # matrix = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
    # print( spiralOrder(matrix) )


    def rotate(A):
        A[:] = [[row[i] for row in A[::-1]] for i in range(len(A))]  #TODO

    # matrix = [[1,2,3],[4,5,6],[7,8,9]]
    # matrix = [[5,1,9,11],[2,4,8,10],[13,3,6,7],[15,14,12,16]]
    # rotate(matrix)


    def luckyNumbers(matrix):
        """
        :type matrix: List[List[int]]
        :rtype: List[int]
        """
        m = len(matrix)
        n = len(matrix[0])
        rowMins = []
        colMaxs = []
        result = []
        # check row mins, col maxs
        # add to result if same index of both lists same
        for i in range(m):
            minOfRow = (10**5)+1
            for j in range(n):
                if matrix[i][j] < minOfRow: minOfRow = matrix[i][j]
            rowMins.append(minOfRow)

        for i in range(n):
            maxOfCol = 0
            for j in range(m):
                if matrix[j][i] > maxOfCol: maxOfCol = matrix[j][i]
            colMaxs.append(maxOfCol)

        for i in range(len(matrix)):
            if rowMins[i] in colMaxs: result.append(rowMins[i])

        return result
    
    # matrix = [[3,7,8],[9,11,13],[15,16,17]]
    # matrix = [[1,10,4,2],[9,3,8,7],[15,16,17,12]]
    # matrix = [[7,8],[1,2]]
    # print( luckyNumbers(matrix) )


    def fizzBuzz(n):
        """
        :type n: int
        :rtype: List[str]
        """
        result = []
        for i in range(1,n+1):  # starts with 1 so have to loop until n+1
            if i % 3 == 0 and i % 5 == 0: val = "FizzBuzz"  # start with most specific
            elif i % 3 == 0: val = "Fizz"
            elif i % 5 == 0: val = "Buzz"
            else: val = str(i)
            result.append(val)
        return result
    
    # print( fizzBuzz(15) )


    def maximumDifference(nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        maxDiff = -1  # default is lowest possible
        n = len(nums)
        if n < 2: return -1  # need at least two elements to compare
        for i in range(n-1):
            for j in range(i+1, n):
                if nums[i] < nums[j] and nums[j] - nums[i] > maxDiff: maxDiff = nums[j] - nums[i]        
        return maxDiff
    
    # nums = [7,1,5,4]
    # nums = [9,4,3,2]
    # nums = [1,5,2,10]
    # print( maximumDifference(nums) )


    def longestConsecutive(nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # sort list, create new list wo dupes, check longest
        if len(nums) == 0: return 0
        if len(nums) == 1: return 1
        nums.sort()  #O(nlogn)
        arr = []
        for i in range(len(nums)):
            if i+1 == len(nums):
                arr.append(nums[i])
                break
            if nums[i] == nums[i+1]: continue
            else: arr.append(nums[i])
        
        count = 1
        longest = 1
        for i in range(len(arr)-1):
            if arr[i] + 1 == arr[i+1]: count += 1
            else: count = 1  # reset
            if count > longest: longest = count
        return longest
    
    # nums = [100,4,200,1,3,2]
    # nums = [0,3,7,2,5,8,4,6,0,1]
    # nums = [0,-1]
    # print( longestConsecutive(nums) )


    def moveZeroes(nums):
        """
        :type nums: List[int]
        :rtype: None Do not return anything, modify nums in-place instead.
        """
        # if 0 remove then append
        for i in range(len(nums)):
            if nums[i] == 0:
                nums.remove(0)
                nums.append(0)
        print(nums)
    
    # nums = [0,1,0,3,12]
    # nums = [0]
    # moveZeroes(nums)


    # BEGAN GITHUB REPO
    