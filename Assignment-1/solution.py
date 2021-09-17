# Authors:  Aaditya Agrawal, 19CS10003
#           Debanjan Saha, 19CS30014
          
import util, model

if __name__ == "__main__":
    file = open("diabetes.csv","r")
    df = util.get_data_from_csv(file)
    X_test, X_train, attributes = util.train_test_split(df)
    root = model.build_decision_tree(X_train,attributes,0,20)
    Y_pred = model.predict_list(root, X_test)
    Y_test = []
    for i in X_test:
        Y_test.append(i["Outcome"])
    
    num = sum(Y_test[i] != Y_pred[i] for i in range(len(Y_test)))
    print((num/len(Y_test))*100)
    dataset, attributes = util.convert_data(df)
    root = model.build_decision_tree(dataset,attributes,0,6)
    print("Making Graphviz Graph")
    util.print_decision_tree(root)
    file.close() 