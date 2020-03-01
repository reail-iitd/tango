from src.GNN.CONSTANTS import *
from src.GNN.models import DGL_GCN, DGL_AE
from src.GNN.dataset_utils import *
import random
import numpy as np
from os import path
from tqdm import tqdm

import torch
import torch.nn as nn

training = "ae" # can be "gcn", "ae"

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

def loss_score(graphs, model):
	criterion = nn.MSELoss()
	total_loss = 0.0
	for iter_num, graph in enumerate(graphs):
		goal_num, world_num, tools, g = graph
		y_pred = model(g)
		if training == 'ae':
			y_true = g.ndata['feat']
		elif training == 'gcn':
			y_true = np.zeros(NUMTOOLS)
			for tool in tools:
				y_true[TOOLS.index(tool)] = 1
			y_true = torch.FloatTensor(y_true.reshape(1,-1))
		loss = criterion(y_pred, y_true)
		total_loss += loss
	return total_loss.item()/len(graphs)

if __name__ == '__main__':
	filename = 'dataset/home_'+str(AUGMENTATION)+'.pkl'
	if path.exists(filename):
		data = pickle.load(open(filename,'rb'))
	else:
		data = DGLDataset("dataset/home/", augmentation=AUGMENTATION)
		pickle.dump(data, open(filename, "wb"))
	train = True
	if train:
		if training == 'gcn':
			model = DGL_GCN(data.features, data.num_objects, 50, len(TOOLS), 3, etypes, nn.functional.relu, 0.5)
		elif training == 'ae':
			model = DGL_AE(data.features, 100, 3, etypes, nn.functional.relu)
		criterion = nn.MSELoss()
		optimizer = torch.optim.Adam(model.parameters() , lr = 0.001)

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
			for iter_num, graph in tqdm(enumerate(train_set)):
				goal_num, world_num, tools, g = graph
				y_pred = model(g)
				if training == 'ae':
					y_true = g.ndata['feat']
				elif training == 'gcn':
					y_true = torch.zeros(NUMTOOLS, dtype=torch.float)
					for tool in tools:
						y_true[TOOLS.index(tool)] = 1
					y_true = y_true.reshape(1,-1)
				loss = criterion(y_pred, y_true)
				# print (loss, loss.shape)
				# loss.backward()
				total_loss += loss
			
			total_loss.backward()
			optimizer.step()
			print (total_loss.item()/len(train_set))

			if (num_epochs % 10 == 0):
				if training == 'gcn':
					print ("Accuracy on training set is ",accuracy_score(data, train_set, model))
					print ("Accuracy on test set is ",accuracy_score(data, test_set, model))
				elif training == 'ae':
					print ("Loss on test set is ", loss_score(test_set, model))
				torch.save(model, MODEL_SAVE_PATH + "/" + model.name + "_" + str(num_epochs) + ".pt")
	else:
		model = torch.load(MODEL_SAVE_PATH + "/400.pt")
	print (accuracy_score(data, data.graphs, model, True))

