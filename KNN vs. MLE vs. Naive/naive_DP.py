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


# Demographic Parity P0[Y=1]: 0.4655688622754491 P1[Y=1] : 0.21686746987951808 # P0[Y=0]: 0.5344311377245509 P1[Y=0]: 0.7831325301204819
