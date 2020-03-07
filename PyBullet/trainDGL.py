from src.GNN.CONSTANTS import *
from src.GNN.models import *
from src.GNN.dataset_utils import *
import random
import numpy as np
from os import path
from tqdm import tqdm

import torch
import torch.nn as nn

training = "agcn-tool" # can be "gcn", "ae", "combined", "agcn", "agcn-tool"
split = "world" # can be "random", "world", "tool"
train = True # can be True or False
globalnode = True # can be True or False
ignoreNoTool = False # can be True or False

def load_dataset(filename):
	global TOOLS, NUMTOOLS
	if not ignoreNoTool: TOOLS.append("no-tool"); NUMTOOLS += 1
	if path.exists(filename):
		return pickle.load(open(filename,'rb'))
	data = DGLDataset("dataset/home/", augmentation=AUGMENTATION, globalNode = globalnode)
	pickle.dump(data, open(filename, "wb"))
	return data

def accuracy_score(dset, graphs, model, modelEnc, verbose = False):
	total_correct = 0
	for graph in graphs:
		goal_num, world_num, tools, g = graph
		if 'gcn' in training:
			y_pred = model(g, goal2vec[goal_num], goalObjects2vec[goal_num])
		elif training == 'combined':
			encoding = modelEnc.encode(g)[-1] if globalnode else modelEnc.encode(g)
			y_pred = model(encoding.flatten(), goal2vec[goal_num], goalObjects2vec[goal_num])
		tools_possible = dset.goal_scene_to_tools[(goal_num,world_num)]
		y_pred = list(y_pred.reshape(-1))
		tool_predicted = TOOLS[y_pred.index(max(y_pred))]
		if tool_predicted in tools_possible:
			total_correct += 1
		elif verbose:
			print (goal_num, world_num, tool_predicted, tools_possible)
	return ((total_correct/len(graphs))*100)

def backprop(optimizer, graphs, model, modelEnc=None):
	total_loss = 0.0
	for iter_num, graph in enumerate(graphs):
		goal_num, world_num, tools, g = graph
		if 'ae' in training:
			y_pred = model(g)
			y_true = g.ndata['feat']
		elif 'gcn' in training:
			y_pred = model(g, goal2vec[goal_num], goalObjects2vec[goal_num])
			y_true = torch.zeros(NUMTOOLS)
			for tool in tools: y_true[TOOLS.index(tool)] = 1
		elif 'combined' in training:
			encoding = modelEnc.encode(g)[-1] if globalnode else modelEnc.encode(g)
			y_pred = model(encoding.flatten(), goal2vec[goal_num], goalObjects2vec[goal_num])
			y_true = torch.zeros(NUMTOOLS)
			for tool in tools: y_true[TOOLS.index(tool)] = 1
		loss = torch.sum((y_pred - y_true)** 2)
		total_loss += loss
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
	print (total_loss.item()/len(train_set))

def backpropGD(optimizer, graphs, model, modelEnc=None):
	total_loss = 0.0
	for iter_num, graph in enumerate(graphs):
		goal_num, world_num, tools, g = graph
		if 'ae' in training:
			y_pred = model(g)
			y_true = g.ndata['feat']
		elif 'gcn' in training:
			y_pred = model(g, goal2vec[goal_num], goalObjects2vec[goal_num])
			y_true = torch.zeros(NUMTOOLS)
			for tool in tools: y_true[TOOLS.index(tool)] = 1
		elif 'combined' in training:
			encoding = modelEnc.encode(g)[-1] if globalnode else modelEnc.encode(g)
			y_pred = model(encoding.flatten(), goal2vec[goal_num], goalObjects2vec[goal_num])
			y_true = torch.zeros(NUMTOOLS)
			for tool in tools: y_true[TOOLS.index(tool)] = 1
		loss = torch.sum((y_pred - y_true)** 2)
		total_loss += loss
	optimizer.zero_grad()
	total_loss.backward()
	optimizer.step()
	print (total_loss.item()/len(train_set))

def random_split(data):
	test_size = int(0.1 * len(data.graphs))
	random.shuffle(data.graphs)
	test_set = data.graphs[:test_size]
	train_set = data.graphs[test_size:]
	return train_set, test_set

def world_split(data):
	test_set = []
	train_set = []
	for i in data.graphs:
		for j in range(1,9):
			if (i[0],i[1]) == (j,j-1):
				test_set.append(i)
				break
		else:
			train_set.append(i)
	return train_set, test_set

def tool_split(data):
	train_set, test_set = world_split(data)
	tool_set, notool_set = [], []
	for graph in train_set:
		if 'no-tool' in graph[2]: notool_set.append(graph)
		else: tool_set.append(graph)
	new_set = []
	for i in range(len(tool_set)-len(notool_set)):
		new_set.append(random.choice(notool_set))
	train_set = tool_set + notool_set + new_set
	return train_set, test_set

if __name__ == '__main__':
	filename = ('dataset/home_'+ 
				("global_" if globalnode else '') + 
				("NoTool_" if not ignoreNoTool else '') + 
				str(AUGMENTATION)+'.pkl')
	data = load_dataset(filename)
	modelEnc = None
	if train:
		if training == 'gcn' and not globalnode:
			model = DGL_GCN(data.features, data.num_objects, GRAPH_HIDDEN, NUMTOOLS, 3, etypes, nn.functional.relu, 0.5)
		elif training == 'gcn' and globalnode:
			model = DGL_GCN_Global(data.features, data.num_objects, GRAPH_HIDDEN, NUMTOOLS, 3, etypes, nn.functional.relu, 0.5)
		elif training == 'ae':
			model = DGL_AE(data.features, GRAPH_HIDDEN, 3, etypes, nn.functional.relu, globalnode)
		elif training == 'combined' and globalnode:
			modelEnc = torch.load("trained_models/GCN-AE_Global_10.pt")#; modelEnc.freeze()
			model = DGL_Decoder_Global(GRAPH_HIDDEN, NUMTOOLS, 3)
		elif training == 'combined' and not globalnode:
			modelEnc = torch.load("trained_models/GCN-AE_10.pt") #; modelEnc.freeze()
			model = DGL_Decoder(GRAPH_HIDDEN, NUMTOOLS, 3)
		elif training == 'agcn':
			# model = torch.load("trained_models/GatedHeteroRGCN_Attention_640_3_Trained.pt")
			model = DGL_AGCN(data.features, data.num_objects, 10 * GRAPH_HIDDEN, NUMTOOLS, 3, etypes, nn.functional.tanh, 0.5)
		elif training == "agcn-tool":
			# model = torch.load("trained_models/GatedHeteroRGCN_Attention_Tool_768_3_Trained.pt")
			model = DGL_AGCN_Tool(data.features, data.num_objects, 12 * GRAPH_HIDDEN, NUMTOOLS, 3, etypes, torch.tanh, 0.5)
		elif training == 'agcn_likelihood':
			model = torch.load("trained_models/GatedHeteroRGCN_Attention_Likelihood128_1_Trained.pt")
			# model = DGL_AGCN_Likelihood(data.features, data.num_objects, 2 * GRAPH_HIDDEN, 1, etypes, torch.tanh, 0.5)
		
		optimizer = torch.optim.Adam(model.parameters() , lr = 0.000001)
		train_set, test_set = world_split(data) if split == 'world' else random_split(data)  if split == 'random' else tool_split(data) 

		print ("Size before split was", len(data.graphs))
		print ("The size of the training set is", len(train_set))
		print ("The size of the test set is", len(test_set))

		for num_epochs in range(NUM_EPOCHS+1):
			random.shuffle(train_set)
			print ("EPOCH " + str(num_epochs))

			backprop(optimizer, train_set, model, modelEnc)

			if (num_epochs % 10 == 0):
				if training != "ae":
					print ("Accuracy on training set is ",accuracy_score(data, train_set, model, modelEnc))
					print ("Accuracy on test set is ",accuracy_score(data, test_set, model, modelEnc, True))
				elif training == 'ae':
					print ("Loss on test set is ", loss_score(test_set, model, modelEnc).item()/len(test_set))
				torch.save(model, MODEL_SAVE_PATH + "/" + model.name + "_" + str(num_epochs) + ".pt")
	else:
		model = torch.load(MODEL_SAVE_PATH + "/400.pt")
	print (accuracy_score(data, data.graphs, model, True))
