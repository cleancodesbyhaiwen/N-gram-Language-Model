import math
import pandas as pd
from pathlib import Path
import numpy as np
import collections

percent = 1
number = int(percent * 4000)

train_set = pd.read_csv(str(Path.cwd()) + "\propublicaTrain.csv")
# X training data
X = train_set.iloc[:number, 1:].values
# Y training data
Y = train_set.iloc[:number, 0].values


# calculate the mahatthan distance between two vector
def manhattan(a, b):
    return sum(abs(val1-val2) for val1, val2 in zip(a,b))

# take x vector as input and {1,0} as output
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

############################
# PP - Predictive Parity
############################

# import test set
test_set = pd.read_csv(str(Path.cwd()) + "\propublicaTest.csv")
X_classify = test_set.iloc[:, 1:].values
X_tmp = test_set.iloc[:, :].values
X_test = []

for i in range(len(X_classify)):
    result = classifier(X_classify[i])
    X_test.append(np.append(X_tmp[i], [int(result)]))

X_test = np.array(X_test)

result_0_race0_Y1 = 0
result_1_race0_Y1 = 0
result_0_race1_Y1 = 0
result_1_race1_Y1 = 0

result_0_race0_Y0 = 0
result_1_race0_Y0 = 0
result_0_race1_Y0 = 0
result_1_race1_Y0 = 0

for i in range(len(X_test)):
    # Y_output = 1, Y_label = 1, A = 0
    if X_test[i][10] == 1 and X_test[i][0] == 1 and X_test[i][3] == 0:
        result_0_race0_Y1 += 1
    # Y_output = 1, Y_label = 0, A = 0
    elif X_test[i][10] == 1 and X_test[i][0] == 0 and X_test[i][3] == 0:
        result_1_race0_Y1 += 1
    # Y_output = 1, Y_label = 1, A = 1
    elif X_test[i][10] == 1 and X_test[i][0] == 1 and X_test[i][3] == 1:
        result_1_race1_Y1 += 1
    # Y_output = 1, Y_label = 0, A = 1
    elif X_test[i][10] == 1 and X_test[i][0] == 0 and X_test[i][3] == 1:
        result_0_race1_Y1 += 1
    # Y_output = 0, Y_label = 1, A = 0
    elif X_test[i][10] == 0 and X_test[i][0] == 1 and X_test[i][3] == 0:
        result_1_race0_Y0 += 1
    # Y_output = 0, Y_label = 0, A = 0
    elif X_test[i][10] == 0 and X_test[i][0] == 0 and X_test[i][3] == 0:
        result_0_race0_Y0 += 1
    # Y_output = 0, Y_label = 1, A = 1
    elif X_test[i][10] == 0 and X_test[i][0] == 1 and X_test[i][3] == 1:
        result_1_race1_Y0 += 1
    # Y_output = 0, Y_label = 0, A = 1
    elif X_test[i][10] == 0 and X_test[i][0] == 0 and X_test[i][3] == 1:
        result_0_race1_Y0 += 1

# total number of data with output 1
Y1_total = result_0_race0_Y1 + result_1_race1_Y1 + result_1_race0_Y1 + result_0_race1_Y1
# P0[Y=1|Y=1]
p_0_1_1 = result_0_race0_Y1 / (result_0_race0_Y1 + result_1_race0_Y1)
# P1[Y=1|Y=1]    
p_1_1_1 = result_1_race1_Y1 / (result_1_race1_Y1 + result_0_race1_Y1)
# P0[Y=0|Y=1]
p_0_0_1 = result_1_race0_Y1 / (result_0_race0_Y1 + result_1_race0_Y1)
# P1[Y=0|Y=1]    
p_1_0_1 = result_0_race1_Y1 / (result_1_race1_Y1 + result_0_race1_Y1)

# total number of data with output 0
Y0_total = result_1_race0_Y0 + result_1_race1_Y0 + result_0_race0_Y0 + result_0_race1_Y0
# P0[Y=1|Y=0]
p_0_1_0 = result_1_race0_Y0 / (result_1_race0_Y0 + result_0_race0_Y0)
# P1[Y=1|Y=0]    
p_1_1_0 = result_1_race1_Y0 / (result_1_race1_Y0 + result_0_race1_Y0)
# P0[Y=0|Y=0]
p_0_0_0 = result_0_race0_Y0 / (result_1_race0_Y0 + result_0_race0_Y0)
# P1[Y=0|Y=0]    
p_1_0_0 = result_0_race1_Y0 / (result_1_race1_Y0 + result_0_race1_Y0)

print("P0[Y=1|Y=1]: " + str(p_0_1_1) + " P1[Y=1|Y=1] : " + str(p_1_1_1) + " P0[Y=0|Y=1]: " + str(p_0_0_1) + " P1[Y=0|Y=1]: " + str(p_1_0_1))
print("P0[Y=1|Y=0]: " + str(p_0_1_0) + " P1[Y=1|Y=0] : " + str(p_1_1_0) + " P0[Y=0|Y=0]: " + str(p_0_0_0) + " P1[Y=0|Y=0]: " + str(p_1_0_0))

"""
P0[Y=1|Y=1]: 0.6421052631578947 P1[Y=1|Y=1] : 0.5398773006134969 P0[Y=0|Y=1]: 0.35789473684210527 P1[Y=0|Y=1]: 0.4601226993865031
P0[Y=1|Y=0]: 0.3681462140992167 P1[Y=1|Y=0] : 0.34530938123752497 P0[Y=0|Y=0]: 0.6318537859007833 P1[Y=0|Y=0]: 0.654690618762475
"""


