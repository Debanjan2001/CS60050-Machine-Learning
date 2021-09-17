class Node:

    def __init__(self, attribute, value):
        self.attribute = attribute
        self.value = value
        self.left = None
        self.right = None

    def details(self):
        if self.left == None and self.right == None:
            return f'{self.attribute} = {self.value}'

        return f'{self.attribute}\n<={self.value}'




    