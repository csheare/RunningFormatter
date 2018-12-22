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

from models.basic_model import Basic_Model
from image_manipulation.image_loader import *

class MLP(Basic_Model):
    def __init__(self, config):

        self.lr = config['model']['lr']
        self.epochs = config['model']['epochs']
        self.n_layers = config['model']['n_layers']
        self.h_units = config['model']['h_units']
        self.act_funcs = config['model']['act_funcs']
        self.batch_size = config['model']['batch_size']
        self.display_step = config['model']['display_step']
        self.image_dimensions = config['model']['image_dimensions']
        self.predict = config['model']['predict']
        self.save = config['model']['save']
        self.num_classes = config['model']['num_classes']
        self.directory = config['model']['directory']
        self.test_image = config['model']['test_image']

    # Create model
    def create_model(self, x, weights, biases):
        # Hidden fully connected layer with 256 neurons
        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        # Hidden fully connected layer with 256 neurons
        layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
        # Output fully connected layer with a neuron for each class
        out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
        return out_layer

    def format_data(self):
        # Import image data
        print("Gathering the data...")

        (X,y) = data_to_numpy(self.directory,self.image_dimensions)
        y =  np.asarray([i.decode("utf-8") for i in y])
        labels = np.unique(y)
        num_examples = get_num_files(self.directory,labels)

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

        return (X_train, X_test,y_train, y_test,num_examples,labels)

    def run(self):
        tf.reset_default_graph()
        num_input = self.image_dimensions[0]*self.image_dimensions[0]*3
        (X_train, X_test,y_train, y_test,num_examples,labels) = self.format_data()

        # tf Graph input
        X = tf.placeholder("float", [None, num_input])
        Y = tf.placeholder("float", [None, self.num_classes])
        n_hidden_1 = self.h_units[0]
        n_hidden_2 = self.h_units[1]
        # Store layers weight & bias
        weights = {
         'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
         'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
         'out': tf.Variable(tf.random_normal([n_hidden_2, self.num_classes]))
        }
        biases = {
         'b1': tf.Variable(tf.random_normal([n_hidden_1])),
         'b2': tf.Variable(tf.random_normal([n_hidden_2])),
         'out': tf.Variable(tf.random_normal([self.num_classes]))
        }

        # Construct model
        logits = self.create_model(X,weights,biases)
        prediction = tf.nn.softmax(logits)

        # Define loss and optimizer
        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=Y))
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
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


        if self.predict:
            saver.restore(sess, '../checkpoints/dataset_mlp')
            print("Model restored.")
            #load predict data
            predict = image_to_numpy(self.test_image,self.image_dimensions)
            predict = np.reshape(predict, [predict.shape[0], -1])
            feed_dict = {X: predict}
            classification = sess.run(prediction, feed_dict)
            max_arg = np.argmax(classification, axis=1)[0]
            print("You are a " + labels[max_arg] + " runner!")
            sess.close()
            return classification

        else:
            # Training cycle
            for epoch in range(self.epochs):
                total_batch = int(num_examples/self.batch_size)
                for step in range(total_batch):
                    batch_x, batch_y = next_batch(self.batch_size, X_train, y_train)
                    # Run optimization op (backprop)
                    sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
                    if step % self.display_step == 0 or step == 1:
                        # Calculate batch loss and accuracy
                        loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                             Y: batch_y})
                        print("Step " + str(step) + ", Minibatch Loss= " + \
                              "{:.4f}".format(loss) + ", Training Accuracy= " + \
                              "{:.3f}".format(acc))
            if self.save:
                saver.save(sess, "../checkpoints/dataset_mlp")

            print("Optimization Finished!")

            # Calculate accuracy
            print("Testing Accuracy:", \
            sess.run(accuracy, feed_dict={X: X_test,
                                          Y: y_test}))

            sess.close()
            return accuracy
