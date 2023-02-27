import math
import pandas as pd
from pathlib import Path
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


# import test set
test_set = pd.read_csv(str(Path.cwd()) + "\propublicaTest.csv")
X_test = test_set.iloc[:, 1:].values
Y_test = test_set.iloc[:, 0].values
# Sensitive Attributes - race
A_test = test_set.iloc[:, 3].values


############################
# EO - Equalized Odds
############################

# Y = 1, A = 0
X_race0_Y1 = X_test[Y_test == 1]
A_test1 = A_test[Y_test == 1]
X_race0_Y1 = X_race0_Y1[A_test1 == 0]
# Y = 0, A = 0
X_race0_Y0 = X_test[Y_test == 0]
A_test2 = A_test[Y_test == 0]
X_race0_Y0 = X_race0_Y0[A_test2 == 0]
# Y = 1, A = 1
X_race1_Y1 = X_test[Y_test == 1]
A_test3 = A_test[Y_test == 1]
X_race1_Y1 = X_race1_Y1[A_test3 == 1]
# Y = 0, A = 1
X_race1_Y0 = X_test[Y_test == 0]
A_test4 = A_test[Y_test == 0]
X_race1_Y0 = X_race1_Y0[A_test4 == 1]

result_0_race0_Y1 = 0
result_1_race0_Y1 = 0
result_0_race1_Y1 = 0
result_1_race1_Y1 = 0
result_0_race0_Y0 = 0
result_1_race0_Y0 = 0
result_0_race1_Y0 = 0
result_1_race1_Y0 = 0

# Y = 1
for i in range(len(X_race0_Y1)):
    result_race0 = classifier(X_race0_Y1[i])
    if result_race0 == 1:
        result_1_race0_Y1 += 1
    else:
        result_0_race0_Y1 += 1

for i in range(len(X_race1_Y1)):
    result_race1 = classifier(X_race1_Y1[i])
    if result_race1 == 1:
        result_1_race1_Y1 += 1
    else:
        result_0_race1_Y1 += 1

# P0[Y=1|Y=1]
p_0_1 = result_1_race0_Y1 / (result_0_race0_Y1 + result_1_race0_Y1)
# P1[Y=1|Y=1]    
p_1_1 = result_1_race1_Y1 / (result_1_race1_Y1 + result_0_race1_Y1)
# P0[Y=0|Y=1]
p_0_0 = result_0_race0_Y1 / (result_0_race0_Y1 + result_1_race0_Y1)
# P1[Y=0|Y=1]    
p_1_0 = result_0_race1_Y1 / (result_1_race1_Y1 + result_0_race1_Y1)

print("EO (Y=1) P0[Y=1|Y=1]: " + str(p_0_1) + " P1[Y=1|Y=1]: " + str(p_1_1) + " P0[Y=0|Y=1]: " + str(p_0_0) + " P1[Y=0|Y=1]: " + str(p_1_0))

# Y = 0
for i in range(len(X_race0_Y0)):
    result_race0 = classifier(X_race0_Y0[i])
    if result_race0 == 1:
        result_1_race0_Y0 += 1
    else:
        result_0_race0_Y0 += 1

for i in range(len(X_race1_Y0)):
    result_race1 = classifier(X_race1_Y0[i])
    if result_race1 == 1:
        result_1_race1_Y0 += 1
    else:
        result_0_race1_Y0 += 1

# P0[Y=1|Y=0]
p_0_1 = result_1_race0_Y0 / (result_1_race0_Y0 + result_0_race0_Y0)
# P1[Y=1|Y=0]    
p_1_1 = result_1_race1_Y0 / (result_1_race1_Y0 + result_0_race1_Y0)
# P0[Y=0|Y=0]
p_0_0 = result_0_race0_Y0 / (result_1_race0_Y0 + result_0_race0_Y0)
# P1[Y=0|Y=0]    
p_1_0 = result_0_race1_Y0 / (result_1_race1_Y0 + result_0_race1_Y0)

print("EO (Y=0) P0[Y=1|Y=0]: " + str(p_0_1) + " P1[Y=1|Y=0]: " + str(p_1_1) + " P0[Y=0|Y=0]: " + str(p_0_0) + " P1[Y=0|Y=0]: " + str(p_1_0))

"""
EO (Y=1) P0[Y=1|Y=1]: 0.5648148148148148 P1[Y=1|Y=1]: 0.3371647509578544 P0[Y=0|Y=1]: 0.4351851851851852 P1[Y=0|Y=1]: 0.6628352490421456
EO (Y=0) P0[Y=1|Y=0]: 0.29651162790697677 P1[Y=1|Y=0]: 0.18610421836228289 P0[Y=0|Y=0]: 0.7034883720930233 P1[Y=0|Y=0]: 0.8138957816377171
"""