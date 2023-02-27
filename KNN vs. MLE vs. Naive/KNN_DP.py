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
# DP - Demograpgic Parity
############################
X_race0 = X_test[A_test == 0]
X_race1 = X_test[A_test == 1]

result_0_race0 = 0
result_1_race0 = 0
result_0_race1 = 0
result_1_race1 = 0
for i in range(len(X_race0)):
    result_race0 = classifier(X_race0[i])
    if result_race0 == 1:
        result_1_race0 += 1
    else:
        result_0_race0 += 1

for i in range(len(X_race1)):
    result_race1 = classifier(X_race1[i])
    if result_race1 == 1:
        result_1_race1 += 1
    else:
        result_0_race1 += 1

# P0[Y=1]
p_0_1 = result_1_race0 / len(X_race0)
# P1[Y=1]    
p_1_1 = result_1_race1 / len(X_race1)
# P0[Y=0]
p_0_0 = result_0_race0 / len(X_race0)
# P1[Y=0]    
p_1_0 = result_0_race1 / len(X_race1)

print("Demographic Parity P0[Y=1]: " + str(p_0_1) + " P1[Y=1] : " + str(p_1_1) + " # P0[Y=0]: " + str(p_0_0) + " P1[Y=0]: " + str(p_1_0))

# Demographic Parity P0[Y=1]: 0.42664670658682635 P1[Y=1] : 0.24548192771084337 # P0[Y=0]: 0.5733532934131736 P1[Y=0]: 0.7545180722891566


