#multilayer perceptron  neural network in a binary classifier
#Used https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/neural_network_raw.py as boilerplate
#Todo make this a class for basic MLP()
import tensorflow as tf
import numpy as np
import sklearn.datasets
import sys
import os
from sklearn.model_selection import train_test_split
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder

sys.path.append(os.path.dirname(os.getcwd()))
sys.path.append(os.getcwd())

from image_manipulation.image_loader import *

# Import image data
print("Gathering the data...")
directory = '/Users/courtneyshearer/Desktop/runners_128'
image_dimensions = [128, 128]

(X,y) = data_to_numpy(directory,image_dimensions)
y =  np.asarray([i.decode("utf-8") for i in y])

(X_train, X_test,y_train, y_test) = train_test_split(X,y, test_size = .3)

print("X_train: %s" % str(X_train.shape))
print("y_train: %s" % str(y_train.shape))
print("X_test: %s" % str(X_test.shape))
print("y_test: %s" % str(y_test.shape))

# One Hot Vector
label_encoder = LabelEncoder()
y_test = label_encoder.fit_transform(y_test)
y_train = label_encoder.fit_transform(y_train)

y_train = keras.utils.to_categorical(y_train,num_classes=2)
y_test = keras.utils.to_categorical(y_test,num_classes=2)

#Flatten X
X_train = np.reshape(X_train, [X_train.shape[0], -1])
X_test = np.reshape(X_test, [X_test.shape[0], -1])

#Parameters
learning_rate = 0
num_steps = 0
batch_size = 0
display_step = 0

# Network Parameters
n_hidden_1 = 256 # 1st layer number of neurons
n_hidden_2 = 256 # 2nd layer number of neurons
num_input = image_dimensions[0] * image_dimensions[1] * 3# data input (img shape: 128*128)
num_classes = 2 # total classes (elite or nonelite)

# tf Graph input
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_classes])

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, num_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}



# Create model
def neural_net(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Construct model
logits = neural_net(X)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

def next_batch(num, data, labels):
    '''
    Return a total of `num` random samples and labels.
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

# Start training
saver = tf.train.Saver()

with tf.Session() as sess:
    print("In Session...")
    # Run the initializer
    sess.run(init)

    for step in range(1, num_steps+1):
        batch_x, batch_y = next_batch(batch_size, X_train, y_train)
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")

    # Calculate accuracy
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: X_test,
                                      Y: y_test}))

    saver.save(sess, "./trained_models/model1.ckpt")

#load predict data
predict = image_to_numpy('/Users/courtneyshearer/Desktop/test_128.jpg',[128,128])
predict = np.reshape(predict, [predict.shape[0], -1])
# v1 = tf.get_variable("v1", shape=[3])

with tf.Session() as sess:
  # Restore variables from disk.
  saver.restore(sess, "./trained_models/model1.ckpt")
  print("Model restored.")

  feed_dict = {X: predict}
  classification = sess.run(prediction, feed_dict)
  classification = [int(i) for i in classification[0]]
  print(classification)
  labels = label_encoder.inverse_transform(classification)
  print(labels)
  runner_type = [labels[index] for index in range(len(classification)) if classification[index] == 1]
  print("You are a " + str(runner_type[0]) + " runner!")
  sess.close()
