import argparse
import tensorflow as tf


def main():
    parser = argparse.ArgumentParser(description='Run Model')
    parser.add_argument('--file', default='', type=str, help='load directory')

    args = parser.parse_args()

    tf.reset_default_graph()
    num_input = image_dimensions[0] * image_dimensions[1] * 3

    #Flatten X
    X_train = np.reshape(X_train, [X_train.shape[0], -1])

    # Create some variables.
    X = tf.placeholder("float", [None, num_input])
    image = tf.get_variable("v1", shape=[3])

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.
with tf.Session() as sess:
   # Restore variables from disk.
   saver.restore(sess, "/tmp/model.ckpt")
   print("Model restored.")
   # Check the values of the variables
   print("v1 : %s" % v1.eval())
   print("v2 : %s" % v2.eval())



if __name__ == '__main__':
    main()
