import numpy as np
import pandas as pd
from pathlib import Path
import math


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

# mean and cov of P[X|Y=0]
mean_0 = np.mean(X_0, axis=0)
cov_0 = np.cov(X_0, rowvar=0)
cov_0 = np.add(cov_0, np.identity(9)*0.3)
# mean and cov of P[X|Y=1]
mean_1 = np.mean(X_1, axis=0)
cov_1 = np.cov(X_1, rowvar=0)
cov_1 = np.add(cov_1, np.identity(9)*0.3)


# P[X|Y=0]
# P[X|Y=1]
# This is the Multivariate Normal Distribution
def multi_gauss_pdf(x, mu, sigma):
    d = len(x)
    det = np.linalg.det(sigma)
    x_mu_diff = np.matrix(x - mu)
    inv = np.linalg.inv(cov_0)  
    denominator = math.pow((2*math.pi),float(d)/2) * math.pow(det,1.0/2)
    numerator = math.pow(math.e, -0.5 * (x_mu_diff * inv * x_mu_diff.T))
    result = numerator / denominator
    return result


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
def classifier(x_input):
    prob_0 = multi_gauss_pdf(x_input,mean_0,cov_0) * p_0
    prob_1 = multi_gauss_pdf(x_input,mean_1,cov_1) * p_1
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
EO (Y=1) P0[Y=1|Y=1]: 0.24691358024691357 P1[Y=1|Y=1]: 0.16475095785440613 P0[Y=0|Y=1]: 0.7530864197530864 P1[Y=0|Y=1]: 0.8352490421455939
EO (Y=0) P0[Y=1|Y=0]: 0.06395348837209303 P1[Y=1|Y=0]: 0.02729528535980149 P0[Y=0|Y=0]: 0.936046511627907 P1[Y=0|Y=0]: 0.9727047146401985
"""