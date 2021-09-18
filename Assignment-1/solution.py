# Authors:  Aaditya Agrawal, 19CS10003
#           Debanjan Saha, 19CS30014
import util, model

def solve_question3(dataframe):

    best_depth, best_accuracy, dtree_root =  util.get_best_depth(
        dataframe=dataframe
    )
    util.print_decision_tree(dtree_root,filename="best_depth_tree.gv")
    print(f"The best depth was found to be = {best_depth}")
    print(f"The accuracy at the above mentioned depth = {best_accuracy}")
    return dtree_root

def solve_question4(root, X_test):
    error_before_pruning = model.get_error(root,X_test) 
    nodes_before_pruning = root.count_nodes()
    util.print_decision_tree(root, "tree1.gv")
    root.prune(
        root = root,
        error = error_before_pruning,
        X_valid = X_test
    )
    error_after_pruning = model.get_error(root, X_test)
    nodes_after_pruning = root.count_nodes()
    util.print_decision_tree(root, "tree2.gv")

    print(f"Before Pruning: {error_before_pruning} || Num Nodes: {nodes_before_pruning}")
    print(f"After Pruning: {error_after_pruning} || Num Nodes: {nodes_after_pruning}")


if __name__ == "__main__":
    file = open("diabetes.csv","r")
    df = util.get_data_from_csv(file)
    file.close() 

    """
    Solution For Question 1-2
    """
    accuracy_from_gini, accuracy_from_gain = 0,0
    best_root, best_accuracy = None, 0
    NUM_LOOPS = 10
    for i in range(0, NUM_LOOPS):
        X_test, X_train, attributes = util.train_test_split(df)
        root_gini = model.build_decision_tree(X_train,attributes, util.calculate_gini_index, 0, 10)
        accuracy_from_gini = model.get_accuracy(root_gini, X_test)
        root_gain = model.build_decision_tree(X_train,attributes,util.calculate_information_gain, 0, 10)
        accuracy_from_gain = model.get_accuracy(root_gain, X_test)

        if accuracy_from_gini > best_accuracy:
            best_accuracy = accuracy_from_gini
            best_root = root_gini
        if accuracy_from_gain > best_accuracy:
            best_accuracy = accuracy_from_gain
            best_root = root_gain
    accuracy_from_gain /= NUM_LOOPS
    accuracy_from_gini /= NUM_LOOPS
    print(accuracy_from_gini, accuracy_from_gain, best_accuracy)

    # print("Making Graphviz Graph")
    # util.print_decision_tree(best_root)
    """
    Solution For Question 1-2
    """

    # """
    # Solution For Question 3
    # """
    # print("\n✨✨✨✨✨✨✨✨✨✨✨✨✨✨ Solving Question 3...")
    # solve_question3(dataframe=df)
    # print("🍰🍰🍰🍰🍰🍰🍰🍰🍰🍰🍰🍰🍰🍰 Finished ... \n")
    # """
    # End of Question 3
    # """

    """
    Solution For Question 4
    """
    print("\n✨✨✨✨✨✨✨✨✨✨✨✨✨✨ Solving Question 4...")
    _,X_test,_ = util.train_test_split(df)
    solve_question4(root = best_root,X_test = X_test)    
    print("🍰🍰🍰🍰🍰🍰🍰🍰🍰🍰🍰🍰🍰🍰 Finished ... \n")
    """
    End of Question 4
    """

    

   