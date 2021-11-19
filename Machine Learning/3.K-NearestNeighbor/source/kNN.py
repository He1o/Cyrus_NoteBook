import numpy

class Node():
    def __init__(self, feature):
        self.father = None
        self.feature = feature
        self.left = None
        self.right = None
        self.divide = None

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
        curNode.divide = dim
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


    def _search(self, root, target):
        if not root:
            return None
        if target[root.divide] < root.feature[root.divide]:
            res = self._search(root.left, target)
        else:
            res = self._search(root.right, target)

        return res if res else root

    def _get_distance(self, x, y):
        return np.sqrt(np.sum((np.array(x) - np.array(y)) ** 2))
    
    def _get_hyper_plane_distance(self, node, target):
        return abs(target[node.divide] - node.feature[node.divide])


    def nearest_neighbour_search(self, root, target):
        nearest_node = self._search(root, target)
        nearest_distance = self._get_distance(nearest_node.feature, target)
        currnode = nearest_node
        
        while currnode != root:
            tempnode = currnode
            currnode = currnode.father  #如果currnode没有父节点，它一定等于root，就不会进入这一层
            if self._get_distance(currnode.feature, target) < nearest_distance:
                nearest_node = currnode
                nearest_distance = self._get_distance(currnode.feature, target)
            
            if 
            

        


pnts = [[2,3], [5,4], [9,6], [4,7], [8,1], [7,2]]
tree = KDTree(pnts)
print(tree)