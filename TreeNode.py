# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

    """ Override print """
    def __str__(self):
        result = ""

        if self is None:
            return
        
        queue = [self]  # queue of nodes
        while queue:
            curr_node = queue.pop(0)
            # if node is not None
            # else log null
            if curr_node is not None: 
                result += str(curr_node.val) + ","
                # log node val and append left/right node iff left/right nodes both not None
                if curr_node.left is None and curr_node.right is None:
                    pass
                else:
                    queue.append(curr_node.left)
                    queue.append(curr_node.right)
            else: 
                result += "null,"

        return result