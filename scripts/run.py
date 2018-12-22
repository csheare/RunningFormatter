'''
Todo: Document Everything
'''

import numpy as np
import sys, argparse
import os
import json

sys.path.append(os.path.dirname(os.getcwd()))
sys.path.append(os.getcwd())

from models.mlp import MLP

def determine_model(type):
    switcher = {
        "MLP" :  MLP(config),
    }
    return switcher.get(type,"nothing")

if __name__ == '__main__':
        parser = argparse.ArgumentParser(description='Parse Arguments')
        parser.add_argument('--config', help='json file containing network specifications', type=str, \
		required=True)
        args = parser.parse_args()
        config = json.load(open(args.config))
        #Todo Add check for type of NN

        print("Determining Model...")
        model = determine_model(config['model']['type'])
        if model == "nothing":
             raise Exception('Please Specify Appropriate Type in Config')

        print("Running Model...")
        model.run()
