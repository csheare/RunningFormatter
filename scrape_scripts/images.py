import json
import os
import ast
from google_scraper import *

def load_json():
    with open("../data/data.json") as f:
        data = json.load(f)
    return data

def load_txt():
    f = open("../data/middlers.txt","r")
    data = []
    for line in f:
        data.append(line.replace("\n",""))
    return data

def create_directory(path):
    try:
        os.mkdir(path)
    except OSError:
        print ("Creation of the directory %s failed" % path)
        raise
    else:
        print ("Successfully created the directory %s " % path)

def main1():
    data = load_json()
    for key,value in data.items():
        print("Searching for..." + str(key))
        path = "../data/images/" + str(key)
        create_directory(path)
        run(key,path,10)

def main2():
    data = load_txt()
    for value in data:
        print("Searching for..." + str(value))
        query = str(value) + " running"
        path = "../data/images/" + str(value)
        try:
             create_directory(path)
        except:
            print("Not Queried")
            continue
        else:
            run(query,path,10)

if __name__ == '__main__':
    main2()
