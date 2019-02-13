import h5py
import numpy as np
import keras
import argparse

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def run(X,y,t):

    (X_train, X_test,y_train, y_test) = train_test_split(X,y, test_size = t)

    print("X_train: %s" % str(X_train.shape))
    print("y_train: %s" % str(y_train.shape))
    print("X_test: %s" % str(X_test.shape))
    print("y_test: %s" % str(y_test.shape))

    #normalize the data between 0 and 1
    X_train = np.divide(X_train,255.0)
    x_test = np.divide(X_test,255.0)

    #create one hot vectors for labels
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.fit_transform(y_test)

    y_train = keras.utils.to_categorical(y_train, num_classes=2)
    y_test = keras.utils.to_categorical(y_test,num_classes=2)


    # create a basic convolutional neural network
    # TODO: add batch norm later
    # VGG 8 and VGG 16 network architecture
    # ResNets : 
    cnn = keras.models.Sequential() 

    cnn.add(keras.layers.Conv2D(64, (3,3), padding="same", activation="relu", input_shape=(128,128,3)))
    cnn.add(keras.layers.MaxPooling2D(2, 2))
    cnn.add(keras.layers.Conv2D(128, (3,3), padding="same", activation="relu"))
    cnn.add(keras.layers.MaxPooling2D(2, 2))
    cnn.add(keras.layers.Conv2D(256, (3,3), padding="same", activation="relu"))
    cnn.add(keras.layers.Flatten())
    cnn.add(keras.layers.Dense(1024, activation="relu"))
    cnn.add(keras.layers.Dense(256, activation="relu"))
    cnn.add(keras.layers.Dense(32, activation="relu"))
    cnn.add(keras.layers.Dense(2, activation="softmax"))
    cnn.summary()

    # compile the model
    cnn.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    # train the model for two epochs, using a batch size of 1
    history = cnn.fit(x=X_train, y=y_train, batch_size=16, epochs=10, validation_split=0.1)
    #plot


    # evaluate the model
    score = cnn.evaluate(x=X_test, y=y_test)

    print("test loss:     %g" % score[0])
    print("test accuracy: %g" % score[1])


def make_tuple(string):
    file = h5py.File(string, "r")
    X = file["runners128/images"]
    y = file["runners128/labels"]
    return (np.asarray(X),np.asarray(y))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="cnn")
    parser.add_argument('--data', type=make_tuple)
    parser.add_argument('--test_size', default=.5, type=float)

    args = parser.parse_args()
    run(args.data[0],args.data[1], args.test_size)
