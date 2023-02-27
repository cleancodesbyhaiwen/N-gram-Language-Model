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

# Select whtether to test on train data or test data
seleted_X = X_train
selected_Y = train_Y

max_leaf = [10,100,1000,10000,100000,1000000,10000000,100000000]
correct = []
wrong = []
for max_num in max_leaf:
    # load, no need to initialize the loaded_rf
    classifier = joblib.load("decisionTree_%d.joblib" % max_num)
    # Accuracy of the classifier
    print('training data acc: ',classifier.score(X_train,train_Y))
    print('testing data acc: ',classifier.score(X_test,test_Y))
    accuracy = classifier.score(seleted_X,selected_Y)
    correct.append(int(accuracy*len(seleted_X)))
    wrong.append(int((1-accuracy)*len(seleted_X)))


# PLot a bar graph

# This part referenced from https://www.geeksforgeeks.org/bar-plot-in-matplotlib/
# set width of bar
barWidth = 0.25
fig,ax = plt.subplots(figsize =(12, 8))
plt.title("1-0 loss", fontdict=None, loc='center')
 
# Set position of bar on X axis
br1 = np.arange(len(correct))
br2 = [x + barWidth for x in br1]

# Make the plot
b1 = plt.bar(br1, correct, color ='r', width = barWidth,
        edgecolor ='grey', label ='correct')
b2 = plt.bar(br2, wrong, color ='g', width = barWidth,
        edgecolor ='grey', label ='wrong')

# Adding Xticks
plt.xlabel('Maximum # of leaf nodes', fontweight ='bold', fontsize = 15)
plt.ylabel('# of correct/wrong', fontweight ='bold', fontsize = 15)
plt.xticks([r + barWidth for r in range(len(correct))],
        [10,100,1000,10000,100000,1000000,10000000,100000000])
 
# This referenced from https://stackoverflow.com/
# questions/30228069/how-to-display-the-value-of-the-bar-on-each-bar-with-pyplot-barh
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.01*height,
                '%d' % int(height),
                ha='center', va='bottom')

autolabel(b1)
autolabel(b2)

plt.legend()
plt.show()