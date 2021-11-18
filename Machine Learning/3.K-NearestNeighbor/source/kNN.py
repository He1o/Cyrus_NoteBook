import numpy

class Node():
    def __init__(self, feature):
        self.father = None
        self.feature = feature
        self.left = None
        self.right = None

    @property
    def brother(self):
        if self == self.father.left:
            return self.father.right
        else:
            return self.father.left

    def __str__(self):
        return 'feature: {}'.format(self.feature)

class KDTree():
    def __init__(self, points):
        self.root = self.build_tree(points)

    def build_tree(self, points, dim = 0, father = None):
        if not points:
            return None
        
        points = sorted(points, key = lambda x: x[dim])
        mid = len(points) // 2
        curNode = Node(points[mid])
        curNode.father = father
        curNode.left  = self.build_tree(points[:mid], (dim + 1) % len(points[0]), curNode)
        curNode.right = self.build_tree(points[mid + 1:], (dim + 1) % len(points[0]), curNode)
        
        return curNode
    
    def __str__(self):
        def inorder(root, depth = 0):
            if not root:
                return
            ret.append('depth: {}, {}'.format(str(depth), str(root)))
            inorder(root.left, depth + 1)
            inorder(root.right, depth + 1)
        
        ret = []
        inorder(self.root)
        return '\n'.join(ret)



pnts = [[2,3], [5,4], [9,6], [4,7], [8,1], [7,2]]
tree = KDTree(pnts)
print(tree)