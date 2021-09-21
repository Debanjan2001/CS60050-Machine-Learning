import util

class Node:

    def __init__(self, attribute, value, index, impurity = None):
        self.attribute = attribute
        self.value = value
        self.index = index 
        self.impurity = impurity
        self.left = None
        self.right = None
        
    def is_leaf(self):
        return (self.left == None and self.right == None)

    def count_nodes(self):
        if self.is_leaf():
            return 1
        if self.left == None:
            return 1 + self.right.count_nodes()
        if self.right == None:
            return 1 + self.left.count_nodes()

        return 1 + self.left.count_nodes() + self.right.count_nodes()

    def remove_children(self):
        left, right, attribute, value = self.left, self.right, self.attribute, self.value
        self.left = None
        self.right = None
        self.attribute = "Outcome"
        self.value = self.get_majority_leaf_value()
        return left, right, attribute, value

    def restore_children(self, left, right, attribute, value):
        self.left = left
        self.right = right
        self.attribute = attribute
        self.value = value

    def get_majority_leaf_value(self):
        positive,negative = 0, 0
        q = [self]
        while(len(q)>0):
            node = q.pop(0)
            if node.is_leaf():
                if node.value == 1:
                    positive += 1 
                else:
                    negative += 1

            if node.left != None:
                q.append(node.left)
            if node.right != None:
                q.append(node.right)
            
        return (1 if positive >= negative else 0)
    
    def prune(self, root, error, X_valid):
        if self.is_leaf():
            return 1000
        if self.left != None:
            self.left.prune(root, error, X_valid)
        if self.right != None:
            self.right.prune(root, error, X_valid)
        
        left, right, attribute, value = self.remove_children()
        cur_error = get_error(root, X_valid)
        if cur_error >= error or root.count_nodes() <= 20:
            self.restore_children( left, right, attribute, value )
        else:
            error = cur_error
    
    def details(self):
        if self.left == None and self.right == None:
            return f'id = {self.index}\n{self.attribute} = {self.value}'

        return f'id = {self.index}\n{self.attribute} <= {self.value}\n impurity={round(self.impurity, 4)}'

def build_decision_tree(dataset, attributes:list, impurity_function = None, current_height = 0, max_height = 10,idx = 1):  
    '''
        This function is the main build function that handles the construction 
        of the decision tree. It is a recursive function. It implements the ID3 
        algorithm with the help of utility functions

        Parameters
        -----------

        dataset:    Contains the examples on the basis of which the decision tree
                    is constructed. It is a list of dicitonaries with attrbiutes as the keys

        attrbiutes: List of strings, denoting the list of attributes

        current_height: The current height upto which the tree has been built

        max_height: Represents the maximum height upto which the tree will be built 

        impurity_function: Function pointer to the function used for calculating impurity

        Returns
        -----------

        root: Root of the decision tree created
    '''  
    # If the no sample remains
    if len(dataset) == 0:
        return None
    
    # if we have one data only or no attributes left then store the majority answer in the node and
    # consider this as the prediction value for this node also
    if len(dataset) == 1 or len(attributes) == 0 or current_height == max_height:
        return Node("Outcome",sum(i["Outcome"] == 1 for i in dataset) > sum(i["Outcome"] == 0 for i in dataset),idx,None)

    # select the best attribute to be selected for this node
    # with the help of the utility functions
    best_attribute,split_value,impurity_measure = util.get_best_attr(dataset, attributes, impurity_function)
    left_dataset, right_dataset = [], []

    if(best_attribute == None):
        return Node("Outcome",sum(i["Outcome"] == 1 for i in dataset) > sum(i["Outcome"] == 0 for i in dataset),idx,None)

    # Divide the dataset according to the split_value
    for row in dataset:
        if row[best_attribute] <= split_value :
            left_dataset.append(row)
        else:
            right_dataset.append(row)

    attributes.pop(attributes.index(best_attribute))    
    # recursively build the left and the right children of
    # the current node and the return this node as the head
    root = Node(best_attribute,split_value,idx,impurity_measure)
    root.left = build_decision_tree(left_dataset,attributes, impurity_function,current_height +1 ,max_height,2*idx)
    root.right = build_decision_tree(right_dataset, attributes, impurity_function, current_height + 1 ,max_height,2*idx+1)
    
    attributes.append(best_attribute)


    if root.left == None and root.right == None:
        root.attribute = 'Outcome'
        root.value = sum(i["Outcome"] == 1 for i in dataset) > sum(i["Outcome"] == 0 for i in dataset)
        root.impurity = None  
  
    return root

def predict(root, data):
    '''
    This is a recursive function is used for prediction of a singe sample
    We do DFS with the use of attrbiutes values stored in data

    Parameters
    ----------
    root:  This is the root node of the decision tree.
    data:   the dictionary which contains the attribute and thier 
            corresponding valuies. The prediction will be made 
            for this data.
    Returns
    -------
    True/False: It returns whether a person has diabetes or not
    '''

    # if the leaf node is reached thenreturn the prediction
    # stored at this node
    if root.left == None and root.right == None:
        return root.value

    # based on the decision either recurse to the left
    # or the right half until the leaf node is reached
    if data[root.attribute] <= root.value:
        return predict(root.left, data)
    return predict(root.right, data)

def predict_list(root, X_input):
    '''
    This is a function which predicts the output for a list of data points 
    on the basis of the decision tree passed as parameter.
    Parameters
    ----------
    root:    This is the root node of the decision tree.
    X_input: the array of dictionary which contains the input data points
                whose output have to be predicted
    Returns
    -------
    Y_pred: List of predicted values
    '''
    Y_pred = []
    for data in X_input:
        Y_pred.append(predict(root, data))
    return Y_pred

def get_accuracy(root, X_test):
    '''
    This is a function which predicts the output for a list of data points 
    on the basis of the decision tree passed as parameter.
    Parameters
    ----------
    root:   This is the root node of the decision tree.
    X_test: Array of dictionaries. The function predicts the output to 
            the input points and finds the accuracy using the given
            output
    Returns
    -------
    Y_pred: List of predicted values
    '''
    Y_pred = predict_list(root, X_test)

    correct_predicted = 0
    total_elements = len(X_test)
    for i in range(len(X_test)):
        correct_predicted += (Y_pred[i] == X_test[i]["Outcome"])
    
    acc = 100*(correct_predicted/total_elements)
    return acc

def get_error(root, X_test):
    return 100.0 - get_accuracy(root,X_test)