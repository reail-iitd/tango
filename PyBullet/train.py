from CONSTANTS import *
from models import GraphEncoder, Decoder
from dataset_utils import Dataset
import random
import numpy as np

import torch
import torch.nn as nn

def accuracy_score(dset, encoder, decoder, numtop = 2):
	total_correct = 0
	for graph in dset:
		adjacency_matrix, node_states, node_ids, node_names, n_nodes, tools, goal_vec = graph
		goal_vec = [torch.Tensor(goal_vec) for i in range(n_nodes)]
		goal_vec = torch.stack(goal_vec, 0)

		x = encoder(adjacency_matrix, node_states, node_ids)
		x = torch.cat([x, goal_vec], 1)

		y_true = np.zeros(n_nodes)
		for tool in tools:
			y_true[node_names.index(tool)] = 1

		y_pred = decoder(x)
		y_pred = [(y_pred[i], i) for i in range(len(y_pred))]
		y_pred.sort()
		print (y_pred[0])

		for i in range(-1,-numtop-1, -1):
			if (y_true[y_pred[i][1]] == 1):
				total_correct += 1
				break

	print ((total_correct/len(dset))*100)


if __name__ == '__main__':
	data = Dataset("dataset/")
	train = False
	if train:
		encoder = GraphEncoder()
		decoder = Decoder()
		criterion = nn.CrossEntropyLoss()
		optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()) , lr = 0.005)

		for num_epochs in range(NUM_EPOCHS+1):
			random.shuffle(data.graphs)
			print ("EPOCH " + str(num_epochs))
			total_loss = 0.0
			optimizer.zero_grad()
			for iter_num,graph in enumerate(data.graphs):
				# print ("Iter num is " + str(iter_num))
				adjacency_matrix, node_states, node_ids, node_names, n_nodes, tools, goal_vec = graph
				goal_vec = [torch.Tensor(goal_vec) for i in range(n_nodes)]
				goal_vec = torch.stack(goal_vec, 0)

				x = encoder(adjacency_matrix, node_states, node_ids)
				x = torch.cat([x, goal_vec], 1)

				y_true = np.zeros(n_nodes)
				
				for tool in tools:
					y_true[node_names.index(tool)] = 1
				y_true = torch.FloatTensor(y_true)

				y_pred = decoder(x)
				# loss = y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred)
				# loss = -torch.sum(loss)
				# print (y_pred)
				loss = torch.sum((y_pred - y_true)** 2)
				# loss.backward()
				total_loss += loss
			total_loss.backward()
			optimizer.step()
			print (total_loss.item()/len(data.graphs))

			if (num_epochs % 10 == 0 and num_epochs > 100):
				torch.save(encoder, ENCODER_SAVE_PATH + "/" + str(num_epochs) + ".pt")
				torch.save(decoder, DECODER_SAVE_PATH + "/" + str(num_epochs) + ".pt")
	else:
		encoder = torch.load(ENCODER_SAVE_PATH + "/400.pt")
		decoder = torch.load(DECODER_SAVE_PATH + "/400.pt")
	accuracy_score(data.graphs, encoder, decoder)

