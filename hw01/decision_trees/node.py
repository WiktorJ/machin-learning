class Node:
    def __init__(self) -> None:
        self.left = None
        self.right = None
        self.prediction = {}
        self.misclassification_rate = None
        self.cost = 1
        self.column = None
        self.pivot = None

    def is_leaf(self):
        return self.left is None and self.right is None
