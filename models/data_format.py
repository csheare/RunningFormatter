'''

Filename: data_formater.py
Input: take a directory formatted in the following way:
/directory_name
 /class-1
 /class-2


Goals:
use old code like data to numpy, which return X and Y
then save in hdf5


Output: a hdf5 file with structure
mydataset
 train
    class1: numpy array
    class2: numpy array
 test
    class1: numpy array
    class2: numpy array
'''

import sys, os
import h5py
import numpy as np
import skimage.io
import argparse

def get_num_files(directory, classes):
    num_files = 0
    for c in classes:
        num_files += len(os.listdir(directory +"/" + c))
    return num_files

def image_to_numpy(location,image_dimensions):
    X = np.empty((1,image_dimensions[0],image_dimensions[1],3))
    X[0] = skimage.io.imread(location)
    return X

#image dimensions is a list [height,width]
def data_to_numpy(directory, image_dimensions):
    # infer class names from the sub-directory names
    classes = os.listdir(directory)

    if '.DS_Store' in classes:
        classes.remove('.DS_Store')

    #determine the number of samples
    num_samples = get_num_files(directory,classes)

    # initialize empty data array and label array
    X = np.empty((num_samples,image_dimensions[0],image_dimensions[1],3))
    y = np.empty(num_samples, dtype="S10")

    # iterate through sub-directories
    i = 0

    for c in classes:
    # get list of images in class c
        filenames = os.listdir(directory +"/%s" % c)
        filenames = [directory + "/%s/%s" % (c, f) for f in filenames]

        # load each image into numpy array
        for fname in filenames:
            image = skimage.io.imread(fname)
            try:
                X[i] = image
            except ValueError:
                new_img = np.resize(image, (image_dimensions[0],image_dimensions[1],3))
                X[i] = new_img
            y[i] = c
            i += 1

    return (X,y)


def make_hdf5(directory,image_dimensions,filename):
    (X,y) = data_to_numpy(directory, image_dimensions)
    file = h5py.File(filename, "w")
    dataset_name = "runners"+ str(image_dimensions[0])
    images = dataset_name + "/images"
    labels = dataset_name + "/labels"
    file.create_group(dataset_name)
    ids = file.create_dataset(images,data=X)
    lds = file.create_dataset(labels,data=y)

def make_tuple(string):
    str_split = string.split(" ")
    return (int(str_split[0]), int(str_split[1]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='original data directory')
    parser.add_argument('--directory', default='', type=str, help='load directory')
    parser.add_argument('--image_dimensions', default='128 128', type=make_tuple, help='specify the image dimensions')
    parser.add_argument('--new_name', default="data.hdf5",type=str,help="new file name")
    args = parser.parse_args()
    make_hdf5(args.directory, args.image_dimensions, args.new_name)

