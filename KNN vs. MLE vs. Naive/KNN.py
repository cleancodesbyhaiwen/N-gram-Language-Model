import math
import pandas as pd
from pathlib import Path
import collections

percent = 0.01
number = int(percent * 4167)

train_set = pd.read_csv(str(Path.cwd()) + "\propublicaTrain.csv")
# X training data
X = train_set.iloc[:number, 1:].values
# Y training data
Y = train_set.iloc[:number, 0].values


# calculate the mahatthan distance between two vector
def manhattan(a, b):
    return sum(abs(val1-val2) for val1, val2 in zip(a,b))

# take x vector as input and {1,0} as output, K = 6
def classifier(x_input):
    result = -1
    smallest = dict()
    for i in range(len(X)):
        distance = manhattan(x_input, X[i])
        smallest[distance] = Y[i]
    od = collections.OrderedDict(sorted(smallest.items()))
    c0 = 0
    c1 = 0

    i = 0
    for key, val in od.items():
        if val == 1:
            c1 += 1
        else:
            c0 += 1
        i += 1
        if i >= 6:
            break
    if c0 >= c1:
        result = 0
    else:
        result = 1

    return result


# import test set
test_set = pd.read_csv(str(Path.cwd()) + "\propublicaTest.csv")
X_test = test_set.iloc[:, 1:].values
Y_test = test_set.iloc[:, 0].values

# number of correct prediction
correct = 0
# number of wrong prediction
wrong = 0
for i in range(len(X_test)):
    result = classifier(X_test[i])
    if result == Y_test[i]:
        correct += 1
    else:
        wrong += 1
    print (str(i) + '/' + str(len(Y_test)), end="\r")

correct_rate = correct / (correct + wrong)
print("percent of training data: " + str(percent) + " correct: " + str(correct) + " wrong: " + str(wrong) + " Accuracy: " + str(correct_rate))


"""
percent of training data: 0.01 correct: 1179 wrong: 821 Accuracy: 0.5895
percent of training data: 0.1 correct: 1304 wrong: 696 Accuracy: 0.652
percent of training data: 0.5 correct: 1251 wrong: 749 Accuracy: 0.6255
percent of training data: 1 correct: 1287 wrong: 713 Accuracy: 0.6435
"""