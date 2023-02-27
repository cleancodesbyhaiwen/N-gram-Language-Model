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
percent of training data: 0.01 correct: 1182 wrong: 818 Accuracy: 0.591
percent of training data: 0.1 correct: 1207 wrong: 793 Accuracy: 0.6035
percent of training data: 0.5 correct: 1213 wrong: 787 Accuracy: 0.6065
percent of training data: 1 correct: 1239 wrong: 761 Accuracy: 0.6195
"""