import pandas as pd
import random
from model import Node
from graphviz import Digraph

def build_data_from_csv(file):
    dataframe = pd.read_csv(file)
    attributes = list(dataframe.columns)
    num_rows = dataframe.shape[0]
    data = [
        {
            key: dataframe[key][index]
            for key in attributes
        }
        for index in range(num_rows)
    ]
    
    attributes.remove("Outcome")
    return data,attributes

def split_node(dataset, attributes, function = None):

    random.shuffle(attributes)

    best_attribute_choice = None
    best_gini_index = 1.0
    split_value = -1

    num_rows = len(dataset)
    for attribute in attributes:
        print(attribute)
        data = [
            {
                "Value": row[attribute],
                "Outcome":row["Outcome"]
            }
            for row in dataset
        ]

        data.sort(key=lambda item:item["Value"])

        for index in range(num_rows - 1):
            row = data[index]
            left_positive, left_negative, right_positive, right_negative = 0, 0, 0, 0
            
            for i in range(0,index+1):
                if data[i]["Outcome"] == 1:
                    left_positive += 1
                else: 
                    left_negative += 1

            for i in range(index+1,num_rows):
                if data[i]["Outcome"] == 1:
                    right_positive += 1
                else: 
                    right_negative += 1

            assert(left_positive+right_negative+right_positive+left_negative == num_rows)
            
            left_gini_impurity = calculate_gini_index(left_positive,left_negative)
            right_gini_impurity = calculate_gini_index(right_positive,right_negative)

            avg_gini_impurity = ( left_gini_impurity * (index + 1) + right_gini_impurity * (num_rows - index - 1) ) / num_rows

            if avg_gini_impurity < best_gini_index :
                best_gini_index = avg_gini_impurity
                best_attribute_choice = attribute
                split_value = row["Value"]

    return best_attribute_choice,split_value,best_gini_index


def calculate_gini_index(positive,negative):
    total = positive + negative
    print(f'{positive} ,{negative}')
    return 1.0 - (positive*positive + negative*negative) / (total * total) 
    

def build_decision_tree(dataset, attributes:list,current_height = 0, max_height = 10):
    
    if len(dataset) == 0 or len(attributes) == 0:
        return None
    
    if len(dataset) == 1 or current_height>=max_height:
        return Node("Outcome",dataset[0]["Outcome"])

    best_attribute,split_value,gini_index = split_node ( dataset, attributes )

    left_dataset, right_dataset = [], []

    for row in dataset:
        if row[best_attribute] <= split_value :
            left_dataset.append(row)
        else:
            right_dataset.append(row)

    root = Node(best_attribute,split_value)
    
    new_attributes = attributes
    # new_attributes.pop(new_attributes.index(best_attribute))    
    

    left_subtree = build_decision_tree(left_dataset,new_attributes,current_height +1 ,max_height)
    right_subtree = build_decision_tree(right_dataset, new_attributes, current_height + 1 ,max_height)

    root.left, root.right = left_subtree, right_subtree
    print(root.details())
    return root


"""
copy pasted part from siba
"""
def print_decision_tree(root):
    
    '''
    this function prints the ddecision tree graph so created 
    and saves the output in the a pdf file 

    Parameter
    ---------
    dtree:  root node of the decision tree for which the 
            the graph needs top be printed
    '''

    # create a new Digraph
    f = Digraph('Decision Tree', filename='decision_tree.gv')
    f.attr(rankdir='LR', size='1000,500')

    # border of the nodes is set to rectangle shape
    f.attr('node', shape='rectangle')

    # Do a breadth first search and add all the edges
    # in the output graph
    q = [root]  # queue for the bradth first search
    while len(q) > 0:
        node = q.pop(0)
        if node.left != None:
            f.edge(node.details(), node.left.details(), label='True')
        if node.right != None:
            f.edge(node.details(), node.right.details(), label='False')

        if node.left != None:
            q.append(node.left)
        if node.right != None:
            q.append(node.right)

    # save file name :  decision_tree.gv.pdf
    f.render('decision_tree.gv', view=True)



if __name__ == "__main__":
    file = open("diabetes.csv","r")
    dataset,attributes = build_data_from_csv(file)
    root = build_decision_tree(dataset,attributes,0,5)
    print("Making Graphviz Graph")
    print_decision_tree(root)
    file.close() 

