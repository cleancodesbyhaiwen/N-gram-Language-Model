import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
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

max_leaf_nodes = [10, 100, 1000, 10000, 100000, 1000000]
total_parameter = [10000, 100000, 1000000, 10000000, 100000000, 1000000000]
pairs = [[10, 1], [100, 1], [1000, 1], [10000, 1], [10000, 2], \
         [10000, 4], [10000, 8], [10000, 16], [10000, 32], [10000, 64], [10000, 128]]

for i in range(len(pairs)):
    # Construct the random forest model
    randomForestModel = RandomForestClassifier(n_estimators=pairs[i][1], criterion = 'gini',max_leaf_nodes=pairs[i][0])
    # train the model using the training data
    randomForestModel.fit(X_train, train_Y)
    # perform prediction on training data
    predicted = randomForestModel.predict(X_train)

    # Accuracy of the classifier
    print('training data acc: ',randomForestModel.score(X_train,train_Y))
    print('testing data acc: ',randomForestModel.score(X_test,test_Y))

    # save
    joblib.dump(randomForestModel, "randomForest_%d_%d.joblib" % (pairs[i][0], pairs[i][1]))