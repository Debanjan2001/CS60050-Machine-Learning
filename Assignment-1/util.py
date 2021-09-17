# Authors:  Aaditya Agrawal, 19CS10003
#           Debanjan Saha, 19CS30014

import pandas as pd
import random
from model import Node
from graphviz import Digraph
import math


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
    split_point = 0.8*len(X)
    X_train, X_test = X[:split_point], X[split_point:]

    return X_train, X_test, attributes

def get_best_attr(dataset, attributes, function = None):
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

            if (left_size == 0) and (right_size == 0):
                continue
            elif (left_size == 0):
                right_impurity = function(right_array)
                avg_impurity = right_impurity
            elif (right_size == 0):
                left_impurity = function(left_array)
                avg_impurity = left_impurity
            else:           
                left_impurity = function(left_array)
                right_impurity = function(right_array)
                avg_impurity = (left_impurity * (left_size) + right_impurity * (right_size) ) / num_rows

            if avg_impurity < best_index :
                best_index = avg_impurity
                best_attribute_choice = attribute
                split_value = row["Value"]
    if(best_attribute_choice == None):
        print(attributes)
        print(dataset)
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



