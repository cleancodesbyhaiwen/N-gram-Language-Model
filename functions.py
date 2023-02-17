import random

# This is the funtion to calculate P(w1:n | y)
def bigram_pr(sentence, bigram_freq, words_freq):
    result = 1
    for i in range(1,len(sentence)):
        # Laplacian Smoothing
        if sentence[i-1] + ' ' + sentence[i] not in bigram_freq:
            bigram_freq = {k:v+1 for k,v in bigram_freq.items()}
            bigram_freq[sentence[i-1] + ' ' + sentence[i]] = 1
            if sentence[i-1] in words_freq:   
                words_freq[sentence[i-1]] += 1
            else:
                words_freq[sentence[i-1]] = 1
            if sentence[i] in words_freq:            
                words_freq[sentence[i]] += 1
            else:
                words_freq[sentence[i]] = 1
        result *= ((bigram_freq[sentence[i-1] + ' ' + sentence[i]]) / words_freq[sentence[i-1]])

    return result


# This is the funtion to calculate P(w1:n | y)
def trigram_pr(sentence, trigram_freq, words_freq):
    result = 1
    for i in range(2,len(sentence)):
        # Laplacian Smoothing
        if sentence[i-2] + ' ' + sentence[i-1] + ' ' + sentence[i] not in trigram_freq:
            trigram_freq = {k:v+1 for k,v in trigram_freq.items()}
            trigram_freq[sentence[i-2] + ' ' + sentence[i-1] + ' ' + sentence[i]] = 1
            if sentence[i-2] + ' ' + sentence[i-1] in words_freq:   
                words_freq[sentence[i-2] + ' ' + sentence[i-1]] += 1
            else:
                words_freq[sentence[i-2] + ' ' + sentence[i-1]] = 1
            if sentence[i-1] + ' ' + sentence[i] in words_freq:   
                words_freq[sentence[i-1] + ' ' + sentence[i]] += 1
            else:
                words_freq[sentence[i-1] + ' ' + sentence[i]] = 1
        result *= ((trigram_freq[sentence[i-2] + ' ' + sentence[i-1] + ' ' + sentence[i]]) / words_freq[sentence[i-2] + ' ' + sentence[i-1]])

    return result

# Functions for generating a sentence based on bigram model
def bigram_generate_sentence(bigram_freq, words):
    key, value = random.choice(list(words.items()))
    curr_word = key
    curr_score = 0
    sentence = [key]
    while True:
        for key in words:
            if curr_word + ' ' + key in bigram_freq:
                score = (bigram_freq[curr_word + ' ' + key] / 50)**2
                if score > curr_score:
                    curr_score = score
                    next_word = key
        sentence.append(next_word)
        curr_word = next_word
        curr_score = 0
        if len(sentence) == 20:
            break
    return sentence

def bigram_generate_sentence_random(bigram_freq, words):
    key, value = random.choice(list(words.items()))
    curr_word = key
    sentence = [key]
    while True:
        lis = []
        for key in words:
            if curr_word + ' ' + key in bigram_freq:
                lis.append(key)
        next_word = random.choice(lis)
        sentence.append(next_word)
        curr_word = next_word
        curr_score = 0
        if len(sentence) == 20:
            break
    return sentence

# Functions for generating a sentence based on trigram model
def trigram_generate_sentence(trigram_freq, words):
    key, value = random.choice(list(words.items()))
    key = key.split(' ')[0]
    curr_word = key
    curr_score = 0
    sentence = [key]
    while True:
        for key in words:
            if curr_word + ' ' + key in trigram_freq:
                score = (trigram_freq[curr_word + ' ' + key] / 50)**2
                if score > curr_score:
                    curr_score = score
                    next_word = key
        sentence.append(next_word)
        curr_word = next_word
        curr_score = 0
        if len(sentence) >= 20:
            break
    return sentence
    

def trigram_generate_sentence_random(trigram_freq, words):
    key, value = random.choice(list(trigram_freq.items()))
    key = key.split(' ')
    first = key[0]
    second = key[1]
    prev_word = first
    curr_word = second
    sentence = [first, second]
    while True:
        lis = []
        for key in words:
            if prev_word + ' ' + curr_word + ' ' + key in trigram_freq:
                lis.append(key)
        next_word = random.choice(lis)
        sentence.append(next_word)
        prev_word = curr_word
        curr_word = next_word
        if len(sentence) >= 20:
            break
    return sentence
    

