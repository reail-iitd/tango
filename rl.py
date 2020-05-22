from src.GNN.CONSTANTS import *
from src.GNN.rl_models import *
from src.GNN.dataset_utils import *
import random
import numpy as np
from os import path
from tqdm import tqdm
import approx
import pandas as pd

import torch
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable

import warnings
warnings.simplefilter("ignore")

# To run:
# python train.py $domain $training_type $model_name

training = argv[2]  
# can be "action"

model_name = argv[3] 
# can be "GGCN", "GGCN_Metric", "GGCN_Metric_Attn", "GGCN_Metric_Attn_L", "GGCN_Metric_Attn_L_NT",
# "GGCN_Metric_Attn_L_NT_C", "GGCN_Metric_Attn_L_NT_C_W", "Final_Metric", "Final_Attn", "Final_L",
# "Final_C", "Final_W"

# Global constants
globalnode = False # can be True or False
split = "world" # can be "random", "world", "tool"
ignoreNoTool = False # can be True or False
sequence = "seq" in training or "action" in training # can be True or False
weighted = ("_W" in model_name) ^ ("Final" in model_name)
graph_seq_length = 4
num_actions = len(possibleActions)
memory_size = 3000
with open('jsons/embeddings/'+embedding+'.vectors') as handle: e = json.load(handle)

def get_all_possible_actions():
	actions = []
	for a in ["moveTo", "pick"]:
		for obj in all_objects:
			if obj != 'husky': actions.append({'name': a, 'args':[obj]})
	actions.extend([{'name': i, 'args': ['stool']} for i in ["climbUp", "climbDown"]])
	actions.append({'name': 'clean', 'args': ['dirt']})
	for a in ["dropTo", "pushTo", "pickNplaceAonB"]:
		for obj in all_objects:
			for obj2 in all_objects:
				if obj != 'husky' and obj2 != 'husky': actions.append({'name': a, 'args':[obj, obj2]})
	for obj in ['glue', 'tape']:
		actions.append({'name': 'apply', 'args':[obj, 'paper']})
	actions.append({'name': 'stick', 'args': ['paper', 'walls']})
	for obj in all_objects_with_states:
		actions.extend([{'name': 'changeState', 'args':[obj, i]} for i in ['open', 'close']])
	actions.extend([{'name': 'changeState', 'args':['light', i]} for i in ['off']])
	return actions

def load_dataset():
	global TOOLS, NUMTOOLS, globalnode
	filename = ('dataset/'+ domain + '_'+ 
				("global_" if globalnode else '') + 
				("NoTool_" if not ignoreNoTool else '') + 
				("seq_" if sequence else '') + 
				(embedding) +
				str(AUGMENTATION)+'.pkl')
	print(filename)
	if globalnode: etypes.append("Global")
	if path.exists(filename):
		return pickle.load(open(filename,'rb'))
	data = DGLDataset("dataset/" + domain + "/", 
			augmentation=AUGMENTATION, 
			globalNode=globalnode, 
			ignoreNoTool=ignoreNoTool, 
			sequence=sequence,
			embedding=embedding)
	pickle.dump(data, open(filename, "wb"))
	return data

def world_split(data):
	test_set = []
	train_set = []
	counter = 0
	for i in data.graphs:
		for j in range(1,9):
			if (i[0],i[1]) == (j,j):
				test_set.append(i)
				break
		else:
			counter +=1 
			train_set.append(i)
	return train_set, test_set

def split_data(data):
	train_set, test_set = world_split(data) if split == 'world' else random_split(data)  if split == 'random' else tool_split(data) 
	print ("Size before split was", len(data.graphs))
	print ("The size of the training set is", len(train_set))
	print ("The size of the test set is", len(test_set))
	return train_set, test_set

def form_initial_dataset():
	data = load_dataset()
	filename = 'dataset/rl_dataset.pkl'
	if path.exists(filename): 
		with open(filename, 'rb') as f:
			df, init_graphs, test_set = pickle.load(f)
		return data, df, init_graphs, test_set
	train_set, test_set = split_data(data)
	df = pd.DataFrame(columns=['goal_num', 'st', 'at', 'st+1', 'r'])
	init_graphs = []
	for datapoint in tqdm(train_set, ncols=80):
		goal_num, world_num, tools, g, t = datapoint
		actionSeq, graphSeq = g; complete = False
		approx.initPolicy(domain, goal_num, world_num)
		old_graph = graphSeq[0]; init_graphs.append((old_graph, goal_num, world_num))
		for action in actionSeq:
			complete, new_graph, err = approx.execAction(goal_num, action, e); 
			if err == '': 
				df = df.append({'goal_num':goal_num, 'st':old_graph, 'at':action,'st+1':new_graph, 'r':1}, ignore_index=True)
				old_graph = new_graph
	pickle.dump((df, init_graphs, test_set), open(filename, "wb"))
	return data, df, init_graphs, test_set

def run_new_plan(model, init_graphs, all_actions):
	g, goal_num, world_num = init_graphs[np.random.choice(range(len(init_graphs)))]
	approx.initPolicy(domain, goal_num, world_num)
	old_graphs, actions, new_graphs, i = [], [], [], 0
	while True:
		possible_actions = []
		for action in all_actions: 
			if approx.checkActionPossible(goal_num, action, e): possible_actions.append(action)
		probs = model.policy(g, goal2vec[goal_num], goalObjects2vec[goal_num], possible_actions)
		a = np.random.choice(possible_actions, p=probs.detach().numpy())
		complete, new_g, err = approx.execAction(goal_num, action, e);
		old_graphs.append(g); actions.append(a); new_graphs.append(new_g)
		g = new_g; i += 1;
		if complete: r = [1]*len(old_graphs); break
		elif i >= 40: r = [-1]*len(old_graphs); break
		if err != '': r = [-1]*len(old_graphs); break
	return pd.DataFrame({'goal_num':goal_num, 'st':old_graphs, 'at':actions, 'st+1':new_graphs, 'r':r}), r[0]

def updateBuffer(model, init_graphs, all_actions, replay_buffer, num_runs):
	dataframes = []; rewards = []
	for run in tqdm(list(range(num_runs)), ncols=80):
		plan_df, plan_r = run_new_plan(model, init_graphs, all_actions)
		dataframes.append(plan_df); rewards.append(plan_r)
	replay_buffer = pd.concat([replay_buffer]+dataframes, ignore_index=True)
	if replay_buffer.shape[0] > memory_size: replay_buffer = replay_buffer[-1*memory_size:]
	return replay_buffer, sum(rewards)/len(rewards)

def get_training_data(replay_buffer, crowdsource_df, sample_size):
	total_data = pd.concat([replay_buffer, crowdsource_df])
	positive_data = total_data[total_data.r == 1].sample(sample_size)
	# negative_data = total_data[total_data.r == -1].sample(sample_size)
	return positive_data #pd.concat([positive_data, negative_data])

if __name__ == '__main__':
	all_actions = get_all_possible_actions()
	data, crowdsource_df, init_graphs, test_set = form_initial_dataset()
	replay_buffer = pd.DataFrame(columns=['goal_num', 'st', 'at', 'st+1', 'r'])
	model = A2C(data.features, data.num_objects, 2 * GRAPH_HIDDEN, 4, 3, etypes, torch.tanh)
	optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)
	l = nn.MSELoss()

	while True:
		# replay_buffer, avg_r = updateBuffer(model, init_graphs, all_actions, replay_buffer, 1)
		for b in range(10):
			dataset = get_training_data(replay_buffer, crowdsource_df, 10)
			for ind in dataset.index:
				goal_num, g, a, newg, r = dataset['goal_num'][ind], dataset['st'][ind], dataset['at'][ind], dataset['st+1'][ind], dataset['r'][ind]
				pred_val = model.value(g, goal2vec[goal_num], goalObjects2vec[goal_num])
				# pred_val_new = model.value(newg, goal2vec[goal_num], goalObjects2vec[goal_num])
				# prob = model.policy(g, goal2vec[goal_num], goalObjects2vec[goal_num])
				# policy_loss = torch.mean(torch.sum((r + pred_val_new - pred_val) * -torch.log(torch.add(a, 0.0001)), 1))
				value_loss = l(Variable(torch.Tensor(r)), pred_val)
				#loss = torch.add(policy_loss, value_loss)
				loss = value_loss
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				print("Loss =", loss.data, " Value Loss =", value_loss.data)

