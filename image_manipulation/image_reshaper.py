import os
import argparse
from PIL import Image

def main():
    parser = argparse.ArgumentParser(description='Change Images in a directory')
    parser.add_argument('--directory', default='/Users/gene/Downloads/', type=str, help='load directory')
    parser.add_argument('--save_directory', default='/Users/Downloads/test', type=str, help='save directory')
    args = parser.parse_args()

    images = os.listdir(args.directory)
    #images.remove('.DS_Store')

    for i in images:
        file = str(args.directory) + str(i)
        newfile = str(args.save_directory) + str(i) 
        image = Image.open(file)
        new_image = image.resize((128, 128))
        try:
            new_image.save(newfile)
        except OSError:
            print("well")

if __name__ == '__main__':
    main()
