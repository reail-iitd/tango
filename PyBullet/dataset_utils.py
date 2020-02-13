import json
from CONSTANTS import *
import numpy as np
import os
import pickle
from src.datapoint import Datapoint

def get_graph_data(pathToDatapoint):

	datapoint = pickle.load(open(pathToDatapoint, "rb"))
	if "home" in pathToDatapoint:
		graph_data = datapoint.getGraph(world = "home")["graph_0"] #Initial Graph

	#List of the names of the nodes
	node_names = [i["name"] for i in graph_data["nodes"]]
	#The id may not be the same as the position in the list
	node_ids = [i["id"] for i in graph_data["nodes"]]

	n_nodes = len(node_names)

	# The nodes are ordered here according to the order in which they were put into the list
	node_states = np.zeros([n_nodes, N_STATES], dtype=np.int32)
	for i, node in enumerate(graph_data["nodes"]):
		states = node["states"]
		for state in states:
			idx = state2indx[state]
			node_states[i, idx] = 1

	adjacency_matrix = np.zeros([N_EDGES, n_nodes, n_nodes], dtype=np.bool)
	for edge in graph_data["edges"]:
		edge_type = edge["relation"]
		src_id = edge["from"]
		tgt_id = edge["to"]

		edge_type_idx = edge2idx[edge_type]
		src_idx = node_ids.index(src_id)
		tgt_idx = node_ids.index(tgt_id)

		adjacency_matrix[edge_type_idx, src_idx, tgt_idx] = True

	return (adjacency_matrix, node_states, node_ids, node_names, n_nodes)

class Dataset():
	def __init__(self, program_dir):
		graphs = []
		all_files = os.walk(program_dir)
		for path, dirs, files in all_files:
			if (len(files) > 0):
				for file in files:
					file_path = path + "/" + file
					# print (file_path)
					try:
						graphs.append(get_graph_data(file_path))
					except AttributeError:
						print ("Could not read ", file_path)

		self.graphs = graphs



