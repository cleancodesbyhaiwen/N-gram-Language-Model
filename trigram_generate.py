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
for i in range(len(gpt_train)):
    if gpt_train[i] in gpt_words_freq:
        gpt_words_freq[gpt_train[i]] += 1
    else:
        gpt_words_freq[gpt_train[i]] = 1

hum_trigram_freq = {}
for i in range(len(hum_train)-2):
    if str(hum_train[i] + ' ' + hum_train[i+1] + ' ' + hum_train[i+2]) in hum_trigram_freq:
        hum_trigram_freq[hum_train[i] + ' ' + hum_train[i+1] + ' ' + hum_train[i+2]] += 1
    else:
        hum_trigram_freq[hum_train[i] + ' ' + hum_train[i+1] + ' ' + hum_train[i+2]] = 1
hum_words_freq = {}
for i in range(len(hum_train)):
    if hum_train[i] in hum_words_freq:
        hum_words_freq[hum_train[i]] += 1
    else:
        hum_words_freq[hum_train[i]] = 1

print("5 sentences generated from gpt data")
for i in range(5):
    print(trigram_generate_sentence_random(gpt_trigram_freq, gpt_words_freq))
print("5 sentences generated from hum data")
for i in range(5):
    print(trigram_generate_sentence_random(hum_trigram_freq, hum_words_freq))