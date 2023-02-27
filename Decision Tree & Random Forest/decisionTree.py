import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier 
import joblib


# image size is 28*28, there are a total of 60000 images
train_X = np.load('train.npy')
train_Y = np.load('trainlabels.npy')
test_X = np.load('test.npy')
test_Y = np.load('testlabels.npy')

# Converting each image matrix to vector
X_train = []
for i in range(len(train_X)):
    X_train.append(train_X[i].ravel())
X_train = np.array(X_train)

X_test = []
for i in range(len(test_X)):
    X_test.append(test_X[i].ravel())
X_test = np.array(X_test)


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

max_leaf = [10,100,1000, 10000,100000,1000000,10000000,10000000]


for max_num in max_leaf:
    # Construct the random forest model
    decisionTree = DecisionTreeClassifier(max_leaf_nodes=max_num)
    # train the model using the training data
    decisionTree.fit(X_train, train_Y)
    # perform prediction on training data
    predicted = decisionTree.predict(X_train)

    # Accuracy of the classifier
    print('training data acc: ',decisionTree.score(X_train,train_Y))
    print('testing data acc: ',decisionTree.score(X_test,test_Y))

    # save
    joblib.dump(decisionTree, "decisionTree_%d.joblib" % max_num)