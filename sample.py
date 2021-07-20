import nltk
import numpy
import tflearn
import tensorflow
import random
import json
from nltk.stem.lancaster import LancasterStemmer
import pickle

stemmer = LancasterStemmer()

with open("intents.json") as file:
    data = json.load(file)
#tokenize
try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
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

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

# Building the model
# tensorflow.function(jit_compile=True)
 # just to reset the previous settings
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8) # fully connected layer to the neural network and 8 neurons on the hidden
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax") # to get the probabilities for each of the output
net = tflearn.regression(net)

model = tflearn.DNN(net) # DNN is a type of neural network

try:
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch=1000, batch_size = 8, show_metric=True)
    model.save("model.tflearn")

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for m in s_words:
        for i, w in enumerate(words):
            if w == m:
                bag[i] = (1)

    return numpy.array(bag)

def chat():
    print("Talk to the bot (type quit to exit)")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        result = model.predict([bag_of_words(inp, words)])[0]
        result_index = numpy.argmax(result)
        tag = labels[result_index]

        if result[result_index]>0.7:
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg["responses"]

            print(random.choice(responses))
        else:
            print("Try again")
chat()