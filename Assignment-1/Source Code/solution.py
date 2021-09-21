# Authors:  Aaditya Agrawal, 19CS10003
#           Debanjan Saha, 19CS30014
import util, model, time

def solve_question12(df):
    accuracy_from_gini, accuracy_from_gain = 0,0
    total_gini_acc, total_gain_acc = 0,0
    best_root, best_accuracy = None, 0
    train, test, valid = None, None, None
    NUM_LOOPS = 10
    for i in range(0, NUM_LOOPS):
        X_test, X_train, attributes = util.train_test_split(df)
        X_train, X_valid = util.train_valid_split(X_train)
        root_gini = model.build_decision_tree(X_train,attributes, util.calculate_gini_index, 0, 10)
        accuracy_from_gini = model.get_accuracy(root_gini, X_test)
        root_gain = model.build_decision_tree(X_train,attributes,util.calculate_information_gain, 0, 10)
        accuracy_from_gain = model.get_accuracy(root_gain, X_test)

        total_gain_acc += accuracy_from_gain
        total_gini_acc += accuracy_from_gini

        if accuracy_from_gini > best_accuracy:
            best_accuracy = accuracy_from_gini
            best_root = root_gini
            train = X_train
            valid = X_valid
            test = X_test

        if accuracy_from_gain > best_accuracy:
            best_accuracy = accuracy_from_gain
            best_root = root_gain
            train = X_train
            valid = X_valid
            test = X_test

    total_gini_acc /= NUM_LOOPS
    total_gain_acc /= NUM_LOOPS
    print(f"Average accuracy from applying Gini index as measure of impurity: {total_gini_acc}")
    print(f"Average accuracy from applying Information Gain as measure of impurity: {total_gain_acc}")
    return train, test, valid, best_root

def solve_question3(dataframe):

    best_depth, best_accuracy, dtree_root =  util.get_best_depth(
        dataframe=dataframe
    )
    print(f"The best depth was found to be = {best_depth}")
    print(f"The accuracy at the above mentioned depth = {best_accuracy}")
    return dtree_root

def solve_question4(root, X_valid):
    error_before_pruning = model.get_error(root,X_valid) 
    nodes_before_pruning = root.count_nodes()
    root.prune(
        root = root,
        error = error_before_pruning,
        X_valid = X_valid
    )
    error_after_pruning = model.get_error(root, X_valid)
    nodes_after_pruning = root.count_nodes()

    print(f"Accuracy Before Pruning: {100 - error_before_pruning} || Num Nodes: {nodes_before_pruning}")
    print(f"Accuracy After Pruning: {100 - error_after_pruning} || Num Nodes: {nodes_after_pruning}")

    return root


if __name__ == "__main__":
    start_time = time.time()

    file = open("diabetes.csv","r")
    df = util.get_data_from_csv(file)
    file.close() 

    """
    Solution For Question 1-2
    """
    print("\n------------ Q1 and 2 Begin ------------\n")
    train, test, valid, best_root = solve_question12(df)     
    print("\n------------ Q1 and 2 Finished ------------\n")

    """
    Solution For Question 3X_test
    """
    print("\n------------ Q3 Begin ------------\n")
    solve_question3(dataframe=df)
    print("\n------------ Q3 Finished ------------\n")
    """
    End of Question 3
    """

    """
    Solution For Question 4
    """
    print("\n------------ Q4 Begin ------------\n")
    best_root = solve_question4(root = best_root, X_valid = test)    
    print("\n------------ Q4 Finished ------------\n")
    """
    End of Question 4
    """
    """
    Solution for Question 5
    """
    print("\n------------ Q5 Begin ------------\n")
    print("\n Generating the tree...\n")
    util.print_decision_tree(best_root, "pruned_tree.gv")
    print("\n The tree has been generated and is saved at [ pruned_tree.gv ]")
    print("\n------------ Q5 Finished ------------\n")
    print("Total time elapsed: %s seconds" % (time.time() - start_time))

    

   