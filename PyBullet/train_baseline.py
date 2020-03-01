from src.GNN.CONSTANTS import *
from src.GNN.models import GraphEncoder_Decoder
from src.GNN.dataset_utils import Dataset
import random
import numpy as np
import argparse

import torch
import torch.nn as nn

def parse_args():
	parser = argparse.ArgumentParser()

	parser.add_argument("--global_node", action="store_true", help="Should a global node for context aggregation be added", default=True)

	args = parser.parse_args()
	return args

def accuracy_score(dset, graphs, model, verbose = False):
	total_correct = 0
	final_print = []
	for graph in graphs:
		adjacency_matrix, node_states, node_ids, node_names, n_nodes, tools, goal_num, world_num, node_vectors, node_size_and_pos = graph
		goal_vec = np.zeros(NUM_GOALS)
		goal_vec[goal_num - 1] = 1

		y_pred = model(adjacency_matrix, node_states, node_ids, node_vectors, node_size_and_pos, goal_vec)

		tools_possible = dset.goal_scene_to_tools[(goal_num,world_num)]
		y_pred = list(y_pred.reshape(-1))
		tool_predicted = TOOLS[y_pred.index(max(y_pred))]

		if tool_predicted in tools_possible:
			total_correct += 1
			final_print.append((goal_num, world_num, tool_predicted, tools_possible))
		else:
			if verbose:
				print (goal_num, world_num, tool_predicted, tools_possible)
		# y_true = np.zeros(NUMTOOLS)
		# for tool in tools:
		# 	y_true[TOOLS.index(tool)] = 1
		# y_true = torch.FloatTensor(y_true.reshape(-1,1))

		# y_pred = [(y_pred[i], i) for i in range(len(y_pred))]
		# y_pred.sort(reverse=True)
		# print (y_pred[0])

		# for i in range(-1,-numtop-1, -1):
		# 	if (y_true[y_pred[i][1]] == 1):
		# 		total_correct += 1
		# 		break
	print ("---------------")
	for i in final_print:
		print (i)
	return ((total_correct/len(graphs))*100)


if __name__ == '__main__':
	args = parse_args()
	data = Dataset("dataset/home/", args)
	train = False
	if train:
		model = GraphEncoder_Decoder(args)
		criterion = nn.CrossEntropyLoss()
		optimizer = torch.optim.Adam(model.parameters() , lr = 0.005)

		#Random test set generator
		# test_size = int(0.1 * len(data.graphs))
		# random.shuffle(data.graphs)
		# test_set = data.graphs[:test_size]
		# train_set = data.graphs[test_size:]

		test_set = []
		train_set = []
		for i in data.graphs:
			for j in range(1,9):
				if (i[6],i[7]) == (j,j-1):
					test_set.append(i)
					break
			else:
				train_set.append(i)
		print ("Size before split was", len(data.graphs))
		print ("The size of the training set is", len(train_set))
		print ("The size of the test set is", len(test_set))

		for num_epochs in range(NUM_EPOCHS+1):
			random.shuffle(train_set)
			print ("EPOCH " + str(num_epochs))

			total_loss = 0.0
			optimizer.zero_grad()
			for iter_num,graph in enumerate(train_set):
				# print ("Iter num is " + str(iter_num))
				adjacency_matrix, node_states, node_ids, node_names, n_nodes, tools, goal_num, world_num, node_vectors, node_size_and_pos = graph
				goal_vec = np.zeros(NUM_GOALS)
				goal_vec[goal_num - 1] = 1

				y_pred = model(adjacency_matrix, node_states, node_ids, node_vectors, node_size_and_pos, goal_vec)

				y_true = np.zeros(NUMTOOLS)
				for tool in tools:
					y_true[TOOLS.index(tool)] = 1
				y_true = torch.FloatTensor(y_true.reshape(1,-1))

				# loss = y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred)
				# loss = -torch.sum(loss)
				# print (y_pred)
				loss = torch.sum((y_pred - y_true)** 2)
				# print (loss, loss.shape)
				# loss.backward()
				total_loss += loss
			
			total_loss.backward()
			optimizer.step()
			print (total_loss.item()/len(train_set))

			if (num_epochs % 10 == 0):
				print ("Accuracy on training set is ",accuracy_score(data, train_set, model))
				print ("Accuracy on test set is ",accuracy_score(data, test_set, model))
				torch.save(model, MODEL_SAVE_PATH + "/" + str(num_epochs) + ".pt")
				# torch.save(decoder, DECODER_SAVE_PATH + "/" + str(num_epochs) + ".pt")
	else:
		model = torch.load(MODEL_SAVE_PATH + "/best_model_baseline_270_29_2_2020_20_28.pt")
		# decoder = torch.load(DECODER_SAVE_PATH + "/400.pt")
	print (accuracy_score(data, data.graphs, model, True))