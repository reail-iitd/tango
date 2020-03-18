from src.datapoint import *
import pickle
import json
from extract_vectors import load_all_vectors
from copy import deepcopy
from src.GNN.CONSTANTS import TOOLS

directory = {1: "dataset/home/goal2-fruits-cupboard/world_home0/",
			 2: "dataset/home/goal1-milk-fridge/world_home1/",
			 3: "dataset/home/goal3-clean-dirt/",
			 4: "dataset/home/goal6-bottles-dumpster/",
			 5: "dataset/home/goal4-stick-paper/",
			 6: "dataset/home/goal2-fruits-cupboard/",
			 7: "dataset/home/goal8-light-off/",
			 8: "dataset/home/goal6-bottles-dumpster/"}

goal_num = {1:2 ,2:1 ,3:3 ,4:6 ,5:4 ,6:2 ,7:8, 8:6}

tools = {1: ["tray2", "stick"],
		 2: ["no-tool"],
		 3: ["stool", "vacuum", "sponge"],
		 4: ["box", "stool", "no-tool"],
		 5: ["stool", "tape", "stick"],
		 6: ["stool", "stick", "book", "box", "no-tool"],
		 7: ["stick", "no-tool"],
		 8: ["stick", "stool", "no-tool", "tray", "tray2", "chair"]}

conceptnet = load_all_vectors("jsons/embeddings/conceptnet.txt") # {} #
fasttext = load_all_vectors("jsons/embeddings/fasttext.txt") # {} #
with open('jsons/embeddings/conceptnet.vectors') as handle: ce = json.load(handle)
with open('jsons/embeddings/fasttext.vectors') as handle: fe = json.load(handle)

def formTestData(testnum):
	all_files = os.walk(directory[testnum]); i = 0
	for path, dirs, files in all_files:
		if (len(files) > 0):
			for file in files:
				file_path = path + "/" + file
				with open(file_path, 'rb') as f:
					datapoint = pickle.load(f)
				for e in [(conceptnet, "conceptnet", ce), (fasttext, "fasttext", fe)]:	
					d = {"goal_num": goal_num[testnum], "tools": tools[testnum]} 
					f = open("dataset/test/home/" + e[1] + "/test"+ str(testnum) + "/" + str(i) + ".graph", "w+") 
					enew = deepcopy(e[2])
					if testnum == 3: enew["mop"] = [0] * 300
					elif testnum == 4: enew["box"] = e[0]["crate"] 
					elif testnum == 5: enew["glue"] = [0] * 300
					elif testnum == 6: enew["apple"] = e[0]["guava"] 
					elif testnum == 7: enew["stool"] = e[0]["headphone"] 
					elif testnum == 8: enew["box"] = [0] * 300
					g = datapoint.getGraph(embeddings = enew)
					d["graph_0"] = g["graph_0"]
					d["tool_embeddings"] = [enew[i] for i in TOOLS]
					f.write(json.dumps(d, indent=2)); f.close()
				i += 1

