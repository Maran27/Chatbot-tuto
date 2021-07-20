import nltk
import numpy
import tflearn
import tensorflow
import random
import json
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()

with open("intents.json") as file:
    data = json.load(file)
#tokenize
words = []
labels = []
docs_x = []
docs_y = [] # what intent/tag its a part off

for intent in data["intents"]:
    for pattern in intent["patterns"]: # stemming is nothing but taking each of the patterns or the words and converting them to the root word
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds) # the reason for extending is because its already a list
        docs_x.append(wrds)
        docs_y.append(intent["tag"])

    if intent["tag"] not in labels:
        labels.append(intent["tag"])

# now we are going to stem the words and remove the duplicates (to see how many words that this has seen alrerady or the vocabulary size).
words = [stemmer.stem(w.lower()) for w in words if w not in "?"]
words = sorted(list(set(words))) # set takes all the words and removes the dulpicates, list converts it to a list and sorted sort its

labels = sorted(labels)

# Bag of words that represents all of the words in any given patterns which is used to the model
# bag of words is called as one hot encoded - which will tell how many times the words has occurred by 0 or 1
# usually they do one hot if the word exists 1 or 0.

training = []
output = []

out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []

    wrds = [stemmer.stem(w) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)

training = numpy.array(training)
output = numpy.array(output)
