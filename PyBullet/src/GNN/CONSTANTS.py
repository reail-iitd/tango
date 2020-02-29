import json

STATES = ["Outside", "Inside", "On", "Off", "Close", "Open", "Up", "Down", "Sticky", "Non_Sticky", "Dirty", "Clean", "Grabbed", "Free", "Welded", "Not_Welded", "Drilled", "Not_Drilled", "Driven", "Not_Driven", "Fueled", "Not_Fueled", "Cut", "Not_Cut", "Painted", "Not_Painted"]
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