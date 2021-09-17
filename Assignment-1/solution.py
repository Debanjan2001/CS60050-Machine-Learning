# Authors:  Aaditya Agrawal, 19CS10003
#           Debanjan Saha, 19CS30014
          
import util, model

def solve_question3(dataframe):

    best_depth, best_accuracy, dtree_root =  util.get_best_depth(
        dataframe=dataframe
    )
    util.print_decision_tree(dtree_root)
    print(f"The best depth was found to be = {best_depth}")
    print(f"The accuracy at the above mentioned depth = {best_accuracy}")

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
    """
    Solution For Question 1-2
    """

    """
    Solution For Question 3
    """
    print("\n✨✨✨✨✨✨✨✨✨✨✨✨✨✨ Solving Question 3...")
    solve_question3(dataframe=df)
    print("🍰🍰🍰🍰🍰🍰🍰🍰🍰🍰🍰🍰🍰🍰 Finished ... \n")
    """
    End of Question 3
    """

    # print("Making Graphviz Graph")
    # util.print_decision_tree(root)