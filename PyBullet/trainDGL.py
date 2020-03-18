from src.GNN.CONSTANTS import *
from src.GNN.models import *
from src.GNN.dataset_utils import *
import random
import numpy as np
from os import path
from tqdm import tqdm

import torch
import torch.nn as nn

training = "agcn-likelihood" # can be "gcn", "ae", "combined", "agcn", "agcn-tool", "agcn-likelihood", "sequence"
split = "world" # can be "random", "world", "tool"
train = False # can be True or False
globalnode = False # can be True or False
ignoreNoTool = False # can be True or False
sequence = training == "sequence" # can be True or False
generalization = True

def load_dataset(filename):
	global TOOLS, NUMTOOLS, globalnode
	if not ignoreNoTool: TOOLS.append("no-tool"); NUMTOOLS += 1
	if globalnode: etypes.append("Global")
	if path.exists(filename):
		return pickle.load(open(filename,'rb'))
	data = DGLDataset("dataset/home/", augmentation=AUGMENTATION, globalNode=globalnode, ignoreNoTool=ignoreNoTool, sequence=sequence)
	pickle.dump(data, open(filename, "wb"))
	return data

def gen_data(testnum):
	if testnum == 1: return DGLDataset("dataset/home/goal2-fruits-cupboard/", augmentation=AUGMENTATION, globalNode=globalnode, ignoreNoTool=True, sequence=sequence)
	if testnum == 2: return DGLDataset("dataset/home/", augmentation=AUGMENTATION, globalNode=globalnode, ignoreNoTool=ignoreNoTool, sequence=sequence)

def gen_score(model, filename):
	testData = TestDataset(filename)
	total_correct = 0
	for graph in testData.graphs:
		goal_num, tools, g = graph
		y_pred = model(g, goal2vec[goal_num], goalObjects2vec[goal_num])
		y_pred = list(y_pred.reshape(-1))
		tool_predicted = TOOLS[y_pred.index(max(y_pred))]
		if tool_predicted in tools: total_correct += 1
		# print(tool_predicted, tools)
	return total_correct * 100.0 / len(testData.graphs)

def gen_train(testnum):
	data = gen_data(testnum)
	model = GGCN(data.features, data.num_objects, 4 * GRAPH_HIDDEN, NUMTOOLS, 5, etypes, torch.tanh, 0.5)
	# model = DGL_Simple_Likelihood(data.features, data.num_objects, 4 * GRAPH_HIDDEN, NUMTOOLS, 5, etypes, torch.tanh, 0.5)
	optimizer = torch.optim.Adam(model.parameters() , lr = 0.0001)
	train_set = data.graphs
	for num_epochs in range(NUM_EPOCHS+1):
		random.shuffle(train_set)
		print ("EPOCH " + str(num_epochs))
		backprop(optimizer, train_set, model, data.num_objects, None)
		if (num_epochs % 1 == 0):
			print ("Accuracy on training set is ",accuracy_score(data, train_set, model, None))
			print ("Generalization score ", gen_score(model, "dataset/test/home/test"+str(testnum)+"/0.graph"))
			torch.save(model, MODEL_SAVE_PATH + "/test" + str(testnum) + "/" + model.name + "_" + str(num_epochs) + ".pt")

def accuracy_score(dset, graphs, model, modelEnc, verbose = False):
	total_correct = 0
	for graph in graphs:
		goal_num, world_num, tools, g, t = graph
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

def printPredictions(model, data=None):
	if not data:
		data = DGLDataset("dataset/home/goal2-fruits-cupboard/", 
			augmentation=AUGMENTATION, 
			globalNode=globalnode, 
			ignoreNoTool=ignoreNoTool, 
			sequence=sequence)
	for graph in data.graphs:
		goal_num, world_num, tools, g, t = graph
		if 'gcn' in training:
			y_pred = model(g, goal2vec[goal_num], goalObjects2vec[goal_num])
		elif training == 'combined':
			encoding = modelEnc.encode(g)[-1] if globalnode else modelEnc.encode(g)
			y_pred = model(encoding.flatten(), goal2vec[goal_num], goalObjects2vec[goal_num])
		tools_possible = data.goal_scene_to_tools[(goal_num,world_num)]
		y_pred = list(y_pred.reshape(-1))
		# y_pred[TOOLS.index("box")] = 0
		tool_predicted = TOOLS[y_pred.index(max(y_pred))]
		if tool_predicted == "tray" or tool_predicted == "tray2":
			print(goal_num, world_num, tool_predicted, tools_possible)
		# print(tool_predicted, "\t\t", tools_possible)

def backprop(optimizer, graphs, model, num_objects, modelEnc=None):
	total_loss = 0.0
	for iter_num, graph in enumerate(graphs):
		goal_num, world_num, tools, g, t = graph
		if 'ae' in training:
			y_pred = model(g)
			y_true = g.ndata['feat']
			loss = torch.sum((y_pred - y_true)** 2)
		elif 'gcn' in training:
			y_pred = model(g, goal2vec[goal_num], goalObjects2vec[goal_num])
			y_true = torch.zeros(NUMTOOLS)
			for tool in tools: y_true[TOOLS.index(tool)] = 1
			loss = torch.sum((y_pred - y_true)** 2) #* 100000.0 / t
		elif 'combined' in training:
			encoding = modelEnc.encode(g)[-1] if globalnode else modelEnc.encode(g)
			y_pred = model(encoding.flatten(), goal2vec[goal_num], goalObjects2vec[goal_num])
			y_true = torch.zeros(NUMTOOLS)
			for tool in tools: y_true[TOOLS.index(tool)] = 1
			loss = torch.sum((y_pred - y_true)** 2)
		elif 'sequence' in training:
			actionSeq, graphSeq = g; loss = 0
			for i in range(len(graphSeq)):
				y_pred = model(graphSeq[i], goal2vec[goal_num], goalObjects2vec[goal_num])
				y_true = action2vec(actionSeq[i], num_objects, 4)
				loss += torch.sum((y_pred - y_true)** 2)
		total_loss += loss
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
	print (total_loss.item()/len(graphs))

def backpropGD(optimizer, graphs, model, num_objects, modelEnc=None):
	total_loss = 0.0
	for iter_num, graph in enumerate(graphs):
		goal_num, world_num, tools, g, t = graph
		if 'ae' in training:
			y_pred = model(g)
			y_true = g.ndata['feat']
			loss = torch.sum((y_pred - y_true)** 2)
		elif 'gcn' in training:
			y_pred = model(g, goal2vec[goal_num], goalObjects2vec[goal_num])
			y_true = torch.zeros(NUMTOOLS)
			for tool in tools: y_true[TOOLS.index(tool)] = 1
			loss = torch.sum((y_pred - y_true)** 2)
		elif 'combined' in training:
			encoding = modelEnc.encode(g)[-1] if globalnode else modelEnc.encode(g)
			y_pred = model(encoding.flatten(), goal2vec[goal_num], goalObjects2vec[goal_num])
			y_true = torch.zeros(NUMTOOLS)
			for tool in tools: y_true[TOOLS.index(tool)] = 1
			loss = torch.sum((y_pred - y_true)** 2)
		elif 'sequence' in training:
			actionSeq, graphSeq = g; loss = 0
			for i in range(len(graphSeq)):
				y_pred = model(graphSeq[i], goal2vec[goal_num], goalObjects2vec[goal_num])
				y_true = action2vec(actionSeq[i], num_objects, 4)
				loss += torch.sum((y_pred - y_true)** 2)
		total_loss += loss
	optimizer.zero_grad()
	total_loss.backward()
	optimizer.step()
	print (total_loss.item()/len(graphs))

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
			if (i[0],i[1]) == (j,j):
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
				("seq_" if sequence else '') + 
				str(AUGMENTATION)+'.pkl')
	data = load_dataset(filename)
	modelEnc = None
	if train and not generalization:
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
			# model = torch.load("trained_models/Simple_Attention_Tool_768_3_Trained.pt")
			model = DGL_Simple_Tool(data.features, data.num_objects, 4 * GRAPH_HIDDEN, NUMTOOLS, 5, etypes, torch.tanh, 0.5)
		elif training == 'agcn-likelihood':
			# model = torch.load("trained_models/GGCN_Metric_Attn_L_256_5_5.pt")
			# model = GGCN_Metric_Attn(data.features, data.num_objects, 4 * GRAPH_HIDDEN, NUMTOOLS, 5, etypes, torch.tanh, 0.5)
			model = DGL_Simple_Likelihood(data.features, data.num_objects, 4 * GRAPH_HIDDEN, NUMTOOLS, 5, etypes, torch.tanh, 0.5)
		elif training == 'sequence':
			model = DGL_AGCN_Action(data.features, data.num_objects + 1, 2 * GRAPH_HIDDEN, 4+1, 3, etypes, torch.tanh, 0.5)

		optimizer = torch.optim.Adam(model.parameters() , lr = 0.0001)
		train_set, test_set = world_split(data) if split == 'world' else random_split(data)  if split == 'random' else tool_split(data) 

		print ("Size before split was", len(data.graphs))
		print ("The size of the training set is", len(train_set))
		print ("The size of the test set is", len(test_set))

		for num_epochs in range(NUM_EPOCHS+1):
			random.shuffle(train_set)
			print ("EPOCH " + str(num_epochs))
			backprop(optimizer, train_set, model, data.num_objects, modelEnc)
			if (num_epochs % 1 == 0):
				if training != "ae" and training != "sequence":
					print ("Accuracy on training set is ",accuracy_score(data, train_set, model, modelEnc))
					print ("Accuracy on test set is ",accuracy_score(data, test_set, model, modelEnc, True))
				elif training == 'ae':
					print ("Loss on test set is ", loss_score(test_set, model, modelEnc).item()/len(test_set))
				if num_epochs % 1 == 0:
					torch.save(model, MODEL_SAVE_PATH + "/" + model.name + "_" + str(num_epochs) + ".pt")
	elif not train and not generalization:
		model = torch.load(MODEL_SAVE_PATH + "/Simple_Attention_Likelihood_256_5_Trained.pt")
		# print ("Accuracy on complete set is ",accuracy_score(data, data.graphs, model, modelEnc))
		printPredictions(model)
	else:
		for i in ["GGCN_256_5_Trained", "GGCN_Metric_256_5_Trained", "GGCN_Metric_Attn_256_5_Trained",\
					"GGCN_Metric_Attn_L_256_5_Trained", "GGCN_Mettric_Attn_L_NT_256_5_Trained",\
					"Simple_Attention_Likelihood_256_5_Trained"]:
			model = torch.load(MODEL_SAVE_PATH + "/" + i + ".pt")
			# printPredictions(model, data)
			print(i, gen_score(model, "dataset/test/"))

