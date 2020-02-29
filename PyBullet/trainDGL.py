from src.GNN.CONSTANTS import *
from src.GNN.models import DGL_GCN
from src.GNN.dataset_utils import *
import random
import numpy as np

import torch
import torch.nn as nn

def accuracy_score(dset, graphs, model, verbose = False):
	total_correct = 0
	for graph in graphs:
		goal_num, world_num, tools, g = graph		
		y_pred = model(g)
		tools_possible = dset.goal_scene_to_tools[(goal_num,world_num)]
		y_pred = list(y_pred.reshape(-1))
		tool_predicted = TOOLS[y_pred.index(max(y_pred))]
		if tool_predicted in tools_possible:
			total_correct += 1
		elif verbose:
			print (goal_num, world_num, tool_predicted, tools_possible)
	return ((total_correct/len(graphs))*100)

if __name__ == '__main__':
	data = DGLDataset("dataset/home/")
	train = True
	if train:
		model = DGL_GCN(data.features, data.num_objects, 500, len(TOOLS), 3, etypes, nn.functional.relu, 0.5)
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
				if (i[0],i[1]) == (j,j-1):
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
			for iter_num, graph in enumerate(train_set):
				goal_num, world_num, tools, g = graph
				y_pred = model(g)
				y_true = np.zeros(NUMTOOLS)
				for tool in tools:
					y_true[TOOLS.index(tool)] = 1
				y_true = torch.FloatTensor(y_true.reshape(1,-1))
				loss = torch.sum((y_pred - y_true)** 2)
				# print (loss, loss.shape)
				loss.backward()
				total_loss += loss
			
			# total_loss.backward()
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

