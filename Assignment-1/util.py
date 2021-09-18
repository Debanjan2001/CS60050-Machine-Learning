# Authors:  Aaditya Agrawal, 19CS10003
#           Debanjan Saha, 19CS30014
import pandas as pd
import random
from graphviz import Digraph
import math
import model
from matplotlib import pyplot as plt

def get_data_from_csv(file):
    dataframe = pd.read_csv(file)
    return dataframe

def convert_data(dataframe):
    '''
    This function accepts a pandas dataframe as input,
    and returns it after converting it into an array of dictionaries.

    Parameters
    -----------
    dataframe: A pandas dataframe containing attributes and the outcome

    Returns
    -----------
    data: Array of dictionaries with each element of array corresponding to an example
            and the corresponding dictionary having attribute as the key  
    attributes: List of strings, denoting the list of attributes
    '''
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

def train_test_split(dataframe):
    '''
    This function accepts a pandas dataframe as input,
    converts it into an array of dictionaries then splits
    the dataset into training and test examples using an
    80: 20 split

    Parameters
    -----------
    dataframe: A pandas dataframe containing attributes and the outcome

    Returns
    -----------
    X_train, X_test: The training and test sets in the form array of dictionaries
    attributes: List of strings, denoting the list of attributes
    '''

    X, attributes = convert_data(dataframe)
    random.shuffle(X)
    split_point = int(0.8*len(X))
    X_train, X_test = X[:split_point], X[split_point:]

    return X_train, X_test, attributes

def get_best_attr(dataset, attributes, impurity_function = None):
    random.shuffle(attributes)

    best_attribute_choice = None
    best_index = 2.0
    split_value = -1

    num_rows = len(dataset)
    for attribute in attributes:
        data = [
            {
                "Value": row[attribute],
                "Outcome":row["Outcome"]
            }
            for row in dataset
        ]

        for index in range(num_rows):
            row = data[index]
            left_array ,right_array = [], []
            
            for i in range(num_rows):
                if data[i]["Value"] <= data[index]["Value"]:
                    left_array.append(data[i]["Outcome"])
                else:
                    right_array.append(data[i]["Outcome"])
            left_size =  len(left_array)
            right_size =  len(right_array)

            if (left_size == 0) or (right_size == 0):
                continue
            else:           
                left_impurity = impurity_function(left_array)
                right_impurity = impurity_function(right_array)
                avg_impurity = (left_impurity * (left_size) + right_impurity * (right_size) ) / num_rows

            if avg_impurity < best_index :
                best_index = avg_impurity
                best_attribute_choice = attribute
                split_value = row["Value"]
                
    return best_attribute_choice,split_value,best_index


def calculate_gini_index(data):
    '''
    This function is a helper function which returns
    the gini index of the array passed as argument. 

    Parameters
    ----------
    data: array of numbers whose gini index is to be calculated.

    Returns
    -------
    var: This is the gini index of the numbers that are given as
            input in data
    '''
    positive, negative, total = 0, 0, len(data)
    for i in data:
        if(i == 1):
            positive += 1
        else:
            negative += 1
    return (2*positive*negative)/(total*total) 


def calculate_information_gain(data):
    '''
    This function  is a helper which returns the
    information gain of the array passed as argument. 

    Parameters
    ----------
    data: array of numbers whose information gain is to be calculated.

    Returns
    -------
    var: This is the information gain of the numbers that are given as
            input in data.
    '''
    
    positive, negative, total = 0, 0, len(data)
    for i in data:
        if(i == 1):
            positive += 1
        else:
            negative += 1
    total = positive + negative
    posProb = positive/total
    negProb = negative/total
    if(posProb == 0 and negProb == 0):
        return 0
    if(posProb == 0):
        return -negProb*math.log2(negProb)
    elif(negProb == 0):
        return -posProb*math.log2(posProb)
    return -posProb*math.log2(posProb) - negProb*math.log2(negProb)


def print_decision_tree(root, filename = 'decision_tree.gv'):
    
    '''
    this function prints the decision tree graph so created 
    and saves the output in the a pdf file 

    Parameter
    ---------
    dtree:  root node of the decision tree for which the 
            the graph needs top be printed
    '''

    # create a new Digraph
    f = Digraph('Decision Tree', filename=filename)
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

    # save file name :  {decision_tree}.pdf
    f.render(filename, view=True)


def get_best_depth(dataframe):
    """
    A helper function to determine the best possible depth for the decision tree
    
    Paramters
    -----------
    dataframe : A Pandas Dataframe built from the csv dataset
    X_test    : Test set  

    Returns:
    -----------
    depth     : best depth for which test_dataset performs the best 
    """
    best_depth, best_accuracy, best_tree = 10000, 0, None
    max_depth = 1+int(math.log2(dataframe.shape[0]))
    X_train, X_test, attributes = train_test_split(dataframe)
    
    depthToAccuracy = []
    nodeCountToAccuracy = []

    from tqdm import tqdm
    """
        Progress Bar might be removed for submission files
    """
    for depth in tqdm(range(1,max_depth+1)):

        root = model.build_decision_tree(
            dataset=X_train, 
            attributes=attributes,
            impurity_function = calculate_gini_index,
            current_height = 0,
            max_height = depth
        )    

        accuracy = model.get_accuracy(
            root = root,
            X_test = X_test
        )

        if accuracy>best_accuracy:
            best_depth = depth
            best_accuracy = accuracy
            best_tree = root

        # root = model.build_decision_tree(
        #     dataset=X_train, 
        #     attributes=attributes,
        #     impurity_function = calculate_information_gain,
        #     current_height = 0,
        #     max_height = depth
        # )    

        # accuracy = model.get_accuracy(
        #     root = root,
        #     X_test = X_test
        # )

        # if accuracy>best_accuracy:
        #     best_depth = depth
        #     best_accuracy = accuracy
        #     best_tree = root

        depthToAccuracy.append([depth,accuracy])
        nodeCountToAccuracy.append([root.count_nodes(), accuracy])

    plt.xlabel('Depth')
    plt.ylabel('Accuracy(%)')
    plt.title("Accuracy Vs Depth")
    plt.plot(
        [x for x,_ in depthToAccuracy],
        [y for _,y in depthToAccuracy],
        marker='o',
        color='red'
    )
    plt.savefig('question3_1.pdf')

    nodeCountToAccuracy.sort()
    plt.clf()

    plt.xlabel('No.of Nodes')
    plt.ylabel('Accuracy(%)')
    plt.title("Accuracy Vs No.of Nodes")
    plt.plot(
        [x for x,_ in nodeCountToAccuracy],
        [y for _,y in nodeCountToAccuracy],
        color='red',
        marker='o'
    )
    plt.savefig('question3_2.pdf')

    return best_depth, best_accuracy, best_tree



