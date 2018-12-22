#multilayer perceptron  neural network in a binary classifier
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

class BasicMLP():
    def __init__(self, lr, epochs, n_layers, h_units, \
        act_funcs, batch_size, disp_step, image_dimensions, \
        n_classes, load, save,directory, test_image):

        self.lr = lr
        self.epochs = epochs
        self.n_layers = n_layers
        self.h_units = h_units
        self.act_funcs = act_funcs
        self.batch_size = batch_size
        self.display_step = disp_step
        self.n_input = n_input
        self.image_dimensions = image_dimensions
        self.load = load
        self.save = save
        self.directory = directory
        self.test_image = test_image

    # Create model
    def create_model(self, x, weights, biases):
        # Hidden fully connected layer with 256 neurons
        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        # Hidden fully connected layer with 256 neurons
        layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
        # Output fully connected layer with a neuron for each class
        out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
        return out_layer

    def format_data(self,self.directory,self.num_input):
        # Import image data
        print("Gathering the data...")
        # directory = '/Users/courtneyshearer/Desktop/runners_128'
        # image_dimensions = [128, 128]

        (X,y) = data_to_numpy(directory,image_dimensions)
        y =  np.asarray([i.decode("utf-8") for i in y])
        labels = np.unique(y)

        (X_train, X_test,y_train, y_test) = train_test_split(X,y, test_size = .5)

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

        return (X_train, X_test,y_train, y_test)

    def run(self,dataset,num_examples):
         tf.reset_default_graph()
         n_input = image_dimensions[0]*image_dimensions[0]*3

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

         # Construct model
         logits = multilayer_perceptron(X)
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

        # Create Saver
        saver = tf.train.Saver()
        # Launch the graph
        sess = tf.Session()
        sess.run(init)

        if self.load:
            saver.restore(sess, '../checkpoints/dataset_nn')

        # To calculate the next batch in training
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

        # Training cycle
        for epoch in range(self.epochs):
            total_batch = int(num_examples/self.batch_size)
            for step in range(total_batch):
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
        if self.save:
            saver.save(sess, "../checkpoints/dataset_nn")

        print("Optimization Finished!")

        # Calculate accuracy
        print("Testing Accuracy:", \
            sess.run(accuracy, feed_dict={X: X_test,
                                          Y: y_test}))

        sess.close()
        return accuracy

        def predict(self,self.test_image,self.image_dimensions):
            #load predict data
            predict = image_to_numpy(test_image,image_dimensions)
            predict = np.reshape(predict, [predict.shape[0], -1])

            with tf.Session() as sess:
              # Restore variables from disk.
              saver.restore(sess, "../checkpoints/dataset_nn")
              print("Model restored.")

              feed_dict = {X: predict}
              classification = sess.run(prediction, feed_dict)
              max_arg = np.argmax(classification, axis=1)[0]
              print("You are a " + labels[max_arg] + " runner!")
              sess.close()
