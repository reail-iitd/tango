from src.GNN.CONSTANTS import *
from src.GNN.models import GraphEncoder_Decoder, GraphAttentionEncoder_Decoder
from src.GNN.dataset_utils import Dataset
import random
import numpy as np

import torch
import torch.nn as nn
import argparse

goal_jsons = ["jsons/home_goals/goal1-milk-fridge.json", "jsons/home_goals/goal2-fruits-cupboard.json",\
            "jsons/home_goals/goal3-clean-dirt.json", "jsons/home_goals/goal4-stick-paper.json",\
            "jsons/home_goals/goal5-cubes-box.json", "jsons/home_goals/goal6-bottles-dumpster.json",\
            "jsons/home_goals/goal7-weight-paper.json", "jsons/home_goals/goal8-light-off.json"]
goal_jsons = [json.load(open(i, "r")) for i in goal_jsons]
object2vec = {}
for i in json.load(open("jsons/objects.json", "r"))["objects"]:
	if "vector" in i:
		object2vec[i["name"]] = np.array(i["vector"])

def parse_args():
	parser = argparse.ArgumentParser()

	parser.add_argument("--global_node", action="store_true", help="Should a global node for context aggregation be added")

	args = parser.parse_args()
	return args

def accuracy_score(dset, graphs, model, verbose = False):
	total_correct = 0

	for graph in graphs:
		adjacency_matrix, node_states, node_ids, node_names, n_nodes, tools, goal_num, world_num, node_vectors, node_size_and_pos = graph
		# goal_vec = np.zeros(NUM_GOALS)
		# goal_vec[goal_num - 1] = 1

		# y_pred = model(adjacency_matrix, node_states, node_ids, node_vectors, node_size_and_pos, goal_vec)
		goal_complete_vec = np.array(goal_jsons[goal_num - 1]["goal-vector"])
		goal_objects = goal_jsons[goal_num - 1]["goal-objects"]
		goal_object_vec = np.zeros(300)
		for i in goal_objects:
			goal_object_vec += object2vec[i]
		goal_object_vec /= len(goal_objects)
		
		#Goal bit
		goal_bits = np.zeros((n_nodes, 1))
		for i in goal_objects:
			ind = node_names.index(i)
			goal_bits[ind] = 1

		y_pred = model(adjacency_matrix, node_states, node_ids, node_vectors, node_size_and_pos, goal_object_vec, goal_complete_vec, goal_bits)

		tools_possible = dset.goal_scene_to_tools[(goal_num,world_num)]
		y_pred = list(y_pred.reshape(-1))
		tool_predicted = TOOLS[y_pred.index(max(y_pred))]

		if tool_predicted in tools_possible:
			total_correct += 1
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

	return ((total_correct/len(graphs))*100)


if __name__ == '__main__':
	args = parse_args()
	data = Dataset("dataset/home/", args)
	train = True
	if train:
		model = GraphAttentionEncoder_Decoder(args)
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
				# goal_vec = np.zeros(NUM_GOALS)
				# goal_vec[goal_num - 1] = 1

				# y_pred = model(adjacency_matrix, node_states, node_ids, node_vectors, node_size_and_pos, goal_vec)
				# print (goal_jsons[goal_num - 1])
				goal_complete_vec = np.array(goal_jsons[goal_num - 1]["goal-vector"])
				goal_objects = goal_jsons[goal_num - 1]["goal-objects"]
				goal_object_vec = np.zeros(300)
				for i in goal_objects:
					goal_object_vec += object2vec[i]
				goal_object_vec /= len(goal_objects)
				
				#Goal bit
				goal_bits = np.zeros((n_nodes, 1))
				for i in goal_objects:
					ind = node_names.index(i)
					goal_bits[ind] = 1

				y_pred = model(adjacency_matrix, node_states, node_ids, node_vectors, node_size_and_pos, goal_object_vec, goal_complete_vec, goal_bits)
				
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
		model = torch.load(MODEL_SAVE_PATH + "/400.pt")
		# decoder = torch.load(DECODER_SAVE_PATH + "/400.pt")
	print (accuracy_score(data, data.graphs, model, True))

