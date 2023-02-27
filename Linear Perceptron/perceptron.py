import numpy as np


def dot(K, L):
   if len(K) != len(L):
      return 0

   return sum(i[0] * i[1] for i in zip(K, L))



training_data = [[4,0,1,-1],[0,6,1,1],[0,4,1,-1]]

w = [-1,-1,1]

mistake = 0
while True:
    curr_mistake = 0
    for ele in training_data:
        print("x: " + str(ele[:3]) + " * w: " + str(w) + ' = ' +  str(dot(w,ele[:3])) + " y = " + str(ele[3]))
        if np.sign(dot(w,ele[:3])) != np.sign(ele[3]) or dot(w,ele[:3]) == 0:
            print("mistake")
            mistake += 1
            curr_mistake += 1
            for i in range(len(w)):
                w[i] += ele[3] * ele[i]
    if curr_mistake == 0:
        break

print(w)
print(mistake)
