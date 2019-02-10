import h5py
import numpy
import keras
import argparse

from sklearn.model_selection import train_test_split

def run(X,y,t):
    print("YO")



def make_tuple(string):
    file = h5py.File(string, "r")
    X = file["runners/images"]
    y = file["runners/labels"]
    return (X,y)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="cnn")
    parser.add_argument('--data', type=make_tuple)
    parser.add_argument('--test_size', default=.5, type=float)

    args = parser.parse_args()
    run(args.data[0],args.data[1], args.test_size)
