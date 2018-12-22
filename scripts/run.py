'''
Todo: Document Everything
'''

import numpy as np
import sys, argparse
import os

sys.path.append(os.path.dirname(os.getcwd()))
sys.path.append(os.getcwd())

from models.mlp import BasicMLP

if __name__ == '__main__':
    	parser.add_argument('--config', help='json file containing network specifications', type=str, \
		required=True)
