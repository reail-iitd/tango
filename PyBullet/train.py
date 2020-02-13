from CONSTANTS import *
from models import GraphEncoder, Decoder
from dataset_utils import Dataset
import random
import numpy as np

import torch
import torch.nn as nn

if __name__ == '__main__':
	encoder = GraphEncoder()
	decoder = Decoder()
	data = Dataset("dataset/")
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()) , lr = 0.0003)
	print (len(data.graphs))

	for num_epochs in range(NUM_EPOCHS):
		random.shuffle(data.graphs)
		print ("EPOCH " + str(num_epochs))
		total_loss = 0.0
		for iter_num,graph in enumerate(data.graphs):
			optimizer.zero_grad()
			# print ("Iter num is " + str(iter_num))
			adjacency_matrix, node_states, node_ids, node_names, n_nodes, tools, goal_vec = random.choice(data.graphs)
			goal_vec = [torch.Tensor(goal_vec) for i in range(n_nodes)]
			goal_vec = torch.stack(goal_vec, 0)

			x = encoder(adjacency_matrix, node_states, node_ids)
			x = torch.cat([x, goal_vec], 1)

			y_true = np.zeros(n_nodes)
			
			for tool in tools:
				y_true[node_names.index(tool)] = 1
			y_true = torch.FloatTensor(y_true)

			y_pred = decoder(x)
			loss = y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred)
			loss = -torch.sum(loss)
			loss.backward()
			total_loss += loss.item()
			optimizer.step()
		print (total_loss/len(data.graphs))