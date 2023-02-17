import re 
from functions import *

# rading text into strings
print("Reading text into strings...")
with open('gpt.txt', 'r', encoding='UTF-8') as f:
    gpt = f.read().replace('\n', '')
f.close()
with open('hum.txt', 'r', encoding='UTF-8') as f:
    hum = f.read().replace('\n', '')
f.close()

# Getting rid of all punctuations except , . ? !
print("Removing punctuations...")
gpt = re.sub("[^a-zA-Z0-9 ,?!\.]", "", gpt)
hum = re.sub("[^a-zA-Z0-9 ,?!\.]", "", hum)

# Storing all words in lists
gpt = gpt.split(" ")
hum = hum.split(" ")

# Add * before start of sentence and _ after sentence
print("Adding * and _ as <start> and <end>...")
gpt_new = []
hum_new = []
gpt = list(filter(None, gpt))
hum = list(filter(None, hum))

i = 0
while True:
    if gpt[i][0].isupper():
        gpt_new.append("*")
    gpt_new.append(gpt[i].lower())
    if gpt[i] == '.':
        gpt_new.append("_")
    i += 1
    if i == len(gpt):
         break
i = 0
while True:
    if hum[i][0].isupper():
        hum_new.append("*")
    hum_new.append(hum[i].lower())
    if hum[i] == '.':
        hum_new.append("_")
    i += 1
    if i == len(hum):
         break
    
# Splitting into 90% training set and 10% test set
print("Splitting into training set and test set...")
gpt_len = len(gpt_new)
gpt_train = gpt_new[:int(gpt_len*0.9)]
gpt_test = gpt_new[int(gpt_len*0.9):]
hum_len = len(hum_new)
hum_train = hum_new[:int(hum_len*0.9)]
hum_test = hum_new[int(hum_len*0.9):]

print("Counting trigram frequencies of training set...")
gpt_trigram_freq = {}
for i in range(len(gpt_train)-2):
    if str(gpt_train[i] + ' ' + gpt_train[i+1] + ' ' + gpt_train[i+2]) in gpt_trigram_freq:
        gpt_trigram_freq[gpt_train[i] + ' ' + gpt_train[i+1] + ' ' + gpt_train[i+2]] += 1
    else:
        gpt_trigram_freq[gpt_train[i] + ' ' + gpt_train[i+1] + ' ' + gpt_train[i+2]] = 1
gpt_words_freq = {}
for i in range(len(gpt_train)-1):
    if str(gpt_train[i] + ' ' + gpt_train[i+1]) in gpt_words_freq:
        gpt_words_freq[gpt_train[i] + ' ' + gpt_train[i+1]] += 1
    else:
        gpt_words_freq[gpt_train[i] + ' ' + gpt_train[i+1]] = 1

hum_trigram_freq = {}
for i in range(len(hum_train)-2):
    if str(hum_train[i] + ' ' + hum_train[i+1] + ' ' + hum_train[i+2]) in hum_trigram_freq:
        hum_trigram_freq[hum_train[i] + ' ' + hum_train[i+1] + ' ' + hum_train[i+2]] += 1
    else:
        hum_trigram_freq[hum_train[i] + ' ' + hum_train[i+1] + ' ' + hum_train[i+2]] = 1
hum_words_freq = {}
for i in range(len(hum_train)-1):
    if str(hum_train[i] + ' ' + hum_train[i+1]) in hum_words_freq:
        hum_words_freq[hum_train[i] + ' ' +  hum_train[i+1]] += 1
    else:
        hum_words_freq[hum_train[i] + ' ' +  hum_train[i+1]] = 1

print("Counting trigram frequencies of test set...")
gpt_trigram_freq_test = {}
for i in range(len(gpt_test)-2):
    if str(gpt_test[i] + ' ' + gpt_test[i+1] + ' ' + gpt_test[i+2]) in gpt_trigram_freq_test:
        gpt_trigram_freq_test[gpt_test[i] + ' ' + gpt_test[i+1] + ' ' + gpt_test[i+2]] += 1
    else:
        gpt_trigram_freq_test[gpt_test[i] + ' ' + gpt_test[i+1] + ' ' + gpt_test[i+2]] = 1
gpt_words_freq_test = {}
for i in range(len(gpt_test)-1):
    if str(gpt_test[i] + ' ' + gpt_test[i+1]) in gpt_words_freq_test:
        gpt_words_freq_test[gpt_test[i] + ' ' + gpt_test[i+1]] += 1
    else:
        gpt_words_freq_test[gpt_test[i] + ' ' +  gpt_test[i+1]] = 1

hum_trigram_freq_test = {}
for i in range(len(hum_test)-2):
    if str(hum_test[i] + ' ' + hum_test[i+1] + ' ' + hum_test[i+2]) in hum_trigram_freq_test:
        hum_trigram_freq_test[hum_test[i] + ' ' + hum_test[i+1] + ' ' + hum_test[i+2]] += 1
    else:
        hum_trigram_freq_test[hum_test[i] + ' ' + hum_test[i+1] + ' ' + hum_test[i+2]] = 1
hum_words_freq_test = {}
for i in range(len(hum_test)-1):
    if str(hum_test[i] + ' ' + hum_test[i+1]) in hum_words_freq_test:
        hum_words_freq_test[hum_test[i] + ' ' +  hum_test[i+1]] += 1
    else:
        hum_words_freq_test[hum_test[i] + ' ' +  hum_test[i+1]] = 1

print("Calculating OOV rate...")
oov_gpt = 0
oov_hum = 0
for key in gpt_trigram_freq_test:
    if key not in gpt_trigram_freq:
        oov_gpt += 1
for key in hum_trigram_freq_test:
    if key not in hum_trigram_freq:
        oov_hum += 1

oov_rate_gpt = oov_gpt / len(gpt_trigram_freq_test)
hum_rate_gpt = oov_hum / len(hum_trigram_freq_test)
print("# of oov in gpt test set: " + str(oov_gpt) + " oov rate is: " + str(oov_rate_gpt))
print("# of oov in hum test set: " + str(oov_hum) + " oov rate is: " + str(hum_rate_gpt))



# This calculates the probabilty of hum vs. gpt in the training set
pr_gpt = len(gpt_new) / (len(gpt_new) + len(hum_new))
pr_hum = 1 - pr_gpt

# splitting test data into 10 words sentences
gpt_test_chunks = [gpt_test[x:x+10] for x in range(0, len(gpt_test), 10)]
hum_test_chunks = [hum_test[x:x+10] for x in range(0, len(hum_test), 10)]
gpt_test_chunks = gpt_test_chunks[:50]
hum_test_chunks = hum_test_chunks[:50]

correct = 0
false = 0
# Running classication on gpt test data
print("Running classification on test set...")
for gpt_sentence in gpt_test_chunks:
    print ("correct: " + str(correct) + " false: " + str(false), end="\r")
    pr_from_gpt = pr_gpt * trigram_pr(gpt_sentence, gpt_trigram_freq, gpt_words_freq)
    pr_from_hum = pr_gpt * trigram_pr(gpt_sentence, hum_trigram_freq, hum_words_freq)
    if pr_from_gpt >= pr_from_hum:
        correct += 1
    else:
        false += 1
for hum_sentence in hum_test_chunks:
    print ("correct: " + str(correct) + " false: " + str(false), end="\r")
    pr_from_gpt = pr_gpt * trigram_pr(hum_sentence, gpt_trigram_freq, gpt_words_freq)
    pr_from_hum = pr_gpt * trigram_pr(hum_sentence, hum_trigram_freq, hum_words_freq)
    if pr_from_gpt <= pr_from_hum:
        correct += 1
    else:
        false += 1

correct_rate = correct / (correct + false)
print("The correct rate of the classifier is " + str(correct_rate))
