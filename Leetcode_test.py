import unittest
import Leetcode
import ListNode
# from Leetcode import *

class Leetcode_Test(unittest.TestCase):
    def test_n_1(self):
        exp = 1
        tst = Leetcode.climbStairs(1)
        self.assertEqual(exp,tst)
        # print(exp,tst)

    def test_n_2(self):
        exp = 2
        tst = Leetcode.climbStairs(2)
        self.assertEqual(exp,tst)
        # print(exp,tst)

    def test_n_3(self):
        exp = 3
        tst = Leetcode.climbStairs(3)
        self.assertEqual(exp,tst)
        # print(exp,tst)

    
    def test_insertionSortList(self):
        exp = [1,2,3,4]
        one = ListNode.ListNode(1)
        two = ListNode.ListNode(2)
        three = ListNode.ListNode(3)
        four = ListNode.ListNode(4)
        four.next = two
        two.next = one
        one.next = three
        tst = Leetcode.insertionSortList(four)
        self.assertEqual(exp,tst)


if __name__ == '__main__':
    unittest.main()