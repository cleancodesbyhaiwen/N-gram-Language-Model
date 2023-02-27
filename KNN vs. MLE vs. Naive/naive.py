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
percent of training data: 0.01 correct: 1104 wrong: 896 Accuracy: 0.552
percent of training data: 0.1 correct: 1226 wrong: 774 Accuracy: 0.613
percent of training data: 0.5 correct: 1324 wrong: 676 Accuracy: 0.662
percent of training data: 1 correct: 1353 wrong: 647 Accuracy: 0.6765
"""