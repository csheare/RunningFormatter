import os
import argparse
from PIL import Image


def directory_reshape(directory, save_directory):
        images = os.listdir(directory)
        #images.remove('.DS_Store')

        for i in images:
            file = str(directory) + str(i)
            newfile = str(save_directory) + str(i)
            image_reshape(file,newfile)
            # image = Image.open(file)
            # new_image = image.resize((128, 128))
            # try:
            #     new_image.save(newfile)
            # except OSError:
            #     print("well")

def image_reshape(image,save_image):
        image_to_resize = Image.open(image)
        new_image = image_to_resize.resize((128, 128))
        try:
            new_image.save(save_image)
        except OSError:
            print("well that stinks!")


def main():
    parser = argparse.ArgumentParser(description='Change Images in a directory')
    parser.add_argument('--directory', type=str, help='load directory', required=False)
    parser.add_argument('--save_directory', type=str, help='save directory', required=False)
    parser.add_argument('--image', type=str, help='load file', required=False)
    parser.add_argument('--save_image',type=str, help='save file',required=False)
    args = parser.parse_args()

    if args.directory:
        directory_reshape(args.directory,args.save_directory)

    if args.image:
        image_reshape(args.image, args.save_image)


if __name__ == '__main__':
    main()
