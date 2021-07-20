# # Building the model
# tensorflow.reset_default_graph() # just to reset the previous settings
# net = tflearn.input_data(shape=[None, len(training[0])])
# net = tflearn.fully_connected(net, 8) # fully connected layer to the neural network and 8 neurons on the hidden
# net = tflearn.fully_connected(net, 8)
# net = tflearn.fully_connected(net, len(output[0]), activation="softmax") # to get the probabilities for each of the output
# net = tflearn.regression(net)
#
# model = tflearn.DNN(net)