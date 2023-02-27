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
P0[Y=1|Y=1]: 0.7843137254901961 P1[Y=1|Y=1] : 0.7962962962962963 P0[Y=0|Y=1]: 0.21568627450980393 P1[Y=0|Y=1]: 0.2037037037037037
P0[Y=1|Y=0]: 0.43109540636042404 P1[Y=1|Y=0] : 0.35737704918032787 P0[Y=0|Y=0]: 0.568904593639576 P1[Y=0|Y=0]: 0.6426229508196721
"""