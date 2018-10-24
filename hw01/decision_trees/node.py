class Node:
    def __init__(self) -> None:
        self.left = None
        self.right = None
        self.prediction = {}
        self.misclassification_rate = None
        self.cost = 1
        self.column = None
        self.pivot = None

    def test_left(self, val):
        return val <= self.pivot
