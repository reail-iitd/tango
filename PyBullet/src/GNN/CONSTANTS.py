import json
import numpy as np
from src.datapoint import embeddings

STATES = ["Outside", "Inside", "On", "Off", "Close", "Open", "Up", "Down", "Sticky", "Non_Sticky", "Dirty", "Clean", "Grabbed", "Free", "Welded", "Not_Welded", "Drilled", "Not_Drilled", "Driven", "Not_Driven", "Fueled", "Not_Fueled", "Cut", "Not_Cut", "Painted", "Not_Painted", "Different_Height", "Same_Height"]
N_STATES = len(STATES)
state2indx = {}
for i,state in enumerate(STATES):
	state2indx[state] = i

EDGES = ["Close", "Inside", "On", "Stuck"]
N_EDGES = len(EDGES)
edge2idx = {}
for i,edge in enumerate(EDGES):
	edge2idx[edge] = i
NUMOBJECTS = len(json.load(open("jsons/objects.json", "r"))["objects"])

PRETRAINED_VECTOR_SIZE = 300
REDUCED_DIMENSION_SIZE = 64
SIZE_AND_POS_SIZE = 10

# EMBEDDING_DIM = 32
N_TIMESEPS = 2
GRAPH_HIDDEN = 64
NUM_EPOCHS = 2000
LOGIT_HIDDEN = 32
NUM_GOALS = 8
# TOOLS = ['stool', 'tray', 'tray2', 'lift', 'ramp', 'big-tray', 'book', 'box', 'chair',\
# 		'stick', 'glue', 'tape', 'mop', 'sponge', 'vacuum', 'no-tool']
TOOLS = ['stool', 'tray', 'tray2', 'lift', 'ramp', 'big-tray', 'book', 'box', 'chair',\
		'stick', 'glue', 'tape', 'mop', 'sponge', 'vacuum']
NUMTOOLS = len(TOOLS)
MODEL_SAVE_PATH = "trained_models/"
AUGMENTATION = 1

# Object to vectors
object2vec = {}; object2idx = {}; idx2object = {}
for i, obj in enumerate(json.load(open("jsons/objects.json", "r"))["objects"]):
	object2vec[obj["name"]] = embeddings[obj["name"]]
	object2idx[obj["name"]] = i
	idx2object[i] = obj["name"]
tool_vec = [object2vec[i] for i in TOOLS]
# Goal objects and vectors
goal_jsons = ["jsons/home_goals/goal1-milk-fridge.json", "jsons/home_goals/goal2-fruits-cupboard.json",\
            "jsons/home_goals/goal3-clean-dirt.json", "jsons/home_goals/goal4-stick-paper.json",\
            "jsons/home_goals/goal5-cubes-box.json", "jsons/home_goals/goal6-bottles-dumpster.json",\
            "jsons/home_goals/goal7-weight-paper.json", "jsons/home_goals/goal8-light-off.json"]
goal2vec, goalObjects2vec, goalObjects = {}, {}, {}
for i in range(len(goal_jsons)):
	goal_json = json.load(open(goal_jsons[i], "r"))
	goal2vec[i+1] = np.array(goal_json["goal-vector"])
	goal_object_vec = np.zeros(300)
	for j in goal_json["goal-objects"]:
		goal_object_vec += object2vec[j]
	goal_object_vec /= len(goal_json["goal-objects"])
	goalObjects[i+1] = goal_json["goal-objects"]
	goalObjects2vec[i+1] = goal_object_vec