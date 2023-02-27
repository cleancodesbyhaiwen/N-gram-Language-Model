import pandas as pd
from pathlib import Path
import numpy as np

percent = 1
number = int(percent * 4167)

train_set = pd.read_csv(str(Path.cwd()) + "\propublicaTrain.csv")
# X training data
X = train_set.iloc[:number, 1:].values
# Y training data
Y = train_set.iloc[:number, 0].values

# X training data with label 0
X_0 = X[Y == 0]
# X training data with label 1
X_1 = X[Y == 1]


unique, counts = np.unique(Y, return_counts=True)
freq = dict(zip(unique, counts))
# number of data with label 0
count_0 = freq[0]
# number of data with label 1
count_1 = freq[1]

# P[Y=0]
p_0 = count_0 / (count_1 + count_0)
# P[Y=1]
p_1 = count_1 / (count_0 + count_1)

# take x vector as input and {1,0} as output
# P[Y] * (Product of P[Xi|Y] i from 1 to 9)
def classifier(x_input):
    prob_0 = 1
    prob_1 = 1
    for i in range(len(x_input)):
        curr_prob_0 = 0
        for j in range(len(X_0)):
            if X_0[j][i] == x_input[i]:
                curr_prob_0 += 1
        curr_prob_0 = curr_prob_0 / count_0
        prob_0 *= curr_prob_0
        curr_prob_1 = 0
        for j in range(len(X_1)):
            if X_1[j][i] == x_input[i]:
                curr_prob_1 += 1
        curr_prob_1 = curr_prob_1 / count_1
        prob_1 *= curr_prob_1
    prob_0 *= p_0
    prob_1 *= p_1
    if prob_0 >= prob_1:
        return 0
    else:
        return 1
    

# import test set
test_set = pd.read_csv(str(Path.cwd()) + "\propublicaTest.csv")
X_test = test_set.iloc[:, 1:].values
Y_test = test_set.iloc[:, 0].values
# Sensitive Attributes - race
A_test = test_set.iloc[:, 3].values

############################
# EO - Equalized Odds
############################

X_race0_Y1 = X_test[Y_test == 1]
A_test1 = A_test[Y_test == 1]
X_race0_Y1 = X_race0_Y1[A_test1 == 0]

X_race0_Y0 = X_test[Y_test == 0]
A_test2 = A_test[Y_test == 0]
X_race0_Y0 = X_race0_Y0[A_test2 == 0]

X_race1_Y1 = X_test[Y_test == 1]
A_test3 = A_test[Y_test == 1]
X_race1_Y1 = X_race1_Y1[A_test3 == 1]

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
EO (Y=1) P0[Y=1|Y=1]: 0.6512345679012346 P1[Y=1|Y=1]: 0.36015325670498083 P0[Y=0|Y=1]: 0.3487654320987654 P1[Y=0|Y=1]: 0.6398467432950191
EO (Y=0) P0[Y=1|Y=0]: 0.29069767441860467 P1[Y=1|Y=0]: 0.12406947890818859 P0[Y=0|Y=0]: 0.7093023255813954 P1[Y=0|Y=0]: 0.8759305210918115
"""