from CONSTANTS import *
from models import GraphEncoder
from dataset_utils import Dataset

import torch

if __name__ == '__main__':
	model = GraphEncoder()
	data = Dataset("dataset/")