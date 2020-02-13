import json

STATES = ["Outside", "Inside", "Open", "Close", "On" , "Off" , "Sticky", "Non_Sticky", "Dirty", "Clean", "Grabbed", "Free"]
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
EMBEDDING_DIM = 32
N_TIMESEPS = 2
GRAPH_HIDDEN = 32
NUM_EPOCHS = 2000
LOGIT_HIDDEN = 16
NUM_GOALS = 8