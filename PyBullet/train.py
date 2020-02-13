from CONSTANTS import *
from models import GraphEncoder
from dataset_utils import Dataset
import random

import torch

if __name__ == '__main__':
	model = GraphEncoder()
	data = Dataset("dataset/home/goal1-milk-fridge/world_home6")

	adjacency_matrix, node_states, node_ids, _ , _ = random.choice(data.graphs)

	x = model(adjacency_matrix, node_states, node_ids)