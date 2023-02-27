import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

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
total_parameter = [10000, 100000, 1000000, 10000000]
pairs = [[10, 1], [100, 1], [1000, 1], [10000, 1], [10000, 2], \
         [10000, 4], [10000, 8], [10000, 16], [10000, 32], [10000, 64], [10000, 128]]
test_error = []
train_error = []
for p in pairs:
    # load, no need to initialize the loaded_rf
    classifier = joblib.load("randomForest_%d_%d.joblib" % (p[0],p[1]))
    # Accuracy of the classifier
    print('training data acc: ',classifier.score(X_train,train_Y))
    print('testing data acc: ',classifier.score(X_test,test_Y))
    train_accuracy = classifier.score(X_train,train_Y)
    test_accuracy = classifier.score(X_test,test_Y)
    train_error.append(1-train_accuracy)
    test_error.append(1-test_accuracy)


# PLot a bar graph

# This part referenced from https://www.geeksforgeeks.org/bar-plot-in-matplotlib/
# set width of bar
barWidth = 0.25
fig,ax = plt.subplots(figsize =(12, 8))
plt.title("Train vs. Test # wrong output for different max # leaf nodes, # estimator pairs", fontdict=None, loc='center')
 
# Set position of bar on X axis
br1 = np.arange(len(test_error))
br2 = [x + barWidth for x in br1]

# Make the plot
b1 = plt.bar(br1, test_error, color ='r', width = barWidth,
        edgecolor ='grey', label ='test error rate')
b2 = plt.bar(br2, train_error, color ='g', width = barWidth,
        edgecolor ='grey', label ='train error rate')

# Adding Xticks
plt.xlabel('(max # leaf, # estimator)', fontweight ='bold', fontsize = 15)
plt.ylabel('Error Rate', fontweight ='bold', fontsize = 15)
plt.xticks([r + barWidth for r in range(len(test_error))],
        [[10, 1], [100, 1], [1000, 1], [10000, 1], [10000, 2], \
         [10000, 4], [10000, 8], [10000, 16], [10000, 32], [10000, 64], [10000, 128]])
 
# This referenced from https://stackoverflow.com/
# questions/30228069/how-to-display-the-value-of-the-bar-on-each-bar-with-pyplot-barh
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.01*height,
                '%.2f' % height,
                ha='center', va='bottom')

autolabel(b1)
autolabel(b2)

plt.legend()
plt.show()