from typing import Coroutine
import util

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

def build_decision_tree(dataset, attributes:list,current_height = 0, max_height = 10, impurity_function = None):  
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
        return Node("Outcome",sum(i["Outcome"] == 1 for i in dataset) > sum(i["Outcome"] == 0 for i in dataset))

    # select the best attribute to be selected for this node
    # with the help of the utility functions
    best_attribute,split_value,gini_index = util.get_best_attr(dataset, attributes, impurity_function)
    left_dataset, right_dataset = [], []

    if(best_attribute == None):
        return Node("Outcome",sum(i["Outcome"] == 1 for i in dataset) > sum(i["Outcome"] == 0 for i in dataset))

    # Divide the dataset according to the split_value
    for row in dataset:
        if row[best_attribute] <= split_value :
            left_dataset.append(row)
        else:
            right_dataset.append(row)

    attributes.pop(attributes.index(best_attribute))    
    # recursively build the left and the right children of
    # the current node and the return this node as the head
    root = Node(best_attribute,split_value)
    root.left = build_decision_tree(left_dataset,attributes,current_height +1 ,max_height, impurity_function)
    root.right = build_decision_tree(right_dataset, attributes, current_height + 1 ,max_height, impurity_function)
    
    attributes.append(best_attribute)


    if root.left == None and root.right == None:
        root.attribute = 'Outcome'
        root.value = sum(i["Outcome"] == 1 for i in dataset) > sum(i["Outcome"] == 0 for i in dataset)  
  
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
    Y_pred = []
    for data in X_input:
        Y_pred.append(predict(root, data))
    return Y_pred

def get_accuracy(root, X_test):
    Y_pred = predict_list(root, X_test)

    correct_predicted = 0
    total_elements = len(X_test)
    for i in range(len(X_test)):
        correct_predicted += (Y_pred[i] == X_test[i]["Outcome"])
    
    acc = 100*(correct_predicted/total_elements)
    return acc