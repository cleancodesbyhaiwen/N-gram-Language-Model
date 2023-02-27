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
P0[Y=1|Y=1]: 0.6784565916398714 P1[Y=1|Y=1] : 0.6527777777777778 P0[Y=0|Y=1]: 0.3215434083601286 P1[Y=0|Y=1]: 0.3472222222222222
P0[Y=1|Y=0]: 0.3165266106442577 P1[Y=1|Y=0] : 0.3211538461538462 P0[Y=0|Y=0]: 0.6834733893557423 P1[Y=0|Y=0]: 0.6788461538461539
"""