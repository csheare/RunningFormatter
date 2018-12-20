'''

    file: image_loader.py
    functions:
    Purpose: This file will load the contents of a directory into a numpy file
    Inspiration: https://github.com/CUFCTL/creative-inquiry/blob/master/assets/notebooks/data-visualization.ipynb

'''

import os
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














# def main():
#     parser = argparse.ArgumentParser(description='Load Images')
#     parser.add_argument('--directory', default='', type=str, help='load directory')
#     parser.add_argument('--np_file', default='', type=str, help='np file created')
#
#     args = parser.parse_args()
#
#
# if __name__ == '__main__':
#     main()
