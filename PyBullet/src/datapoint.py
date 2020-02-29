from copy import deepcopy
from src.utils import *
import json
from tqdm import tqdm

tools = ['stool', 'tray', 'tray2', 'lift', 'ramp', 'big-tray', 'book', 'box', 'chair',\
		'stick', 'glue', 'tape', 'mop', 'sponge', 'vacuum', 'drill', 'screwdriver',\
		'hammer', 'ladder', 'trolley', 'brick', 'blow_dryer']

skip = ['ur5', 'cupboard_back', 'fridge_back', 'ramp_tape']

objects = None
with open('jsons/objects.json', 'r') as handle:
    objects = json.load(handle)['objects']

allStates = None
with open('jsons/states.json', 'r') as handle:
    allStates = json.load(handle)

class Datapoint:
	def __init__(self):
		# World
		self.world = ""
		# Goal
		self.goal = ""
		# Robot position list
		self.position = []
		# Metrics of all objects
		self.metrics = []
		# Sticky objects
		self.sticky = []
		# Fixed objects
		self.fixed = []
		# Has cleaner
		self.cleaner = []
		# Action
		self.actions = []
		# Constraints
		self.constraints = []
		# Symbolic actions
		self.symbolicActions = []
		# Objects on
		self.on = []
		# Objects Cleaned
		self.clean = []
		# Stick with object
		self.stick = []
		# Objects fueled
		self.fueled = []
		# Objects welded
		self.welded = []
		# Objects painted
		self.painted = []
		# Objects drilled
		self.drilled = []
		# Objects cut
		self.cut = []
		# Time
		self.time = 0

	def addPoint(self, pos, sticky, fixed, cleaner, action, cons, metric, on, clean, stick, welded, drilled, painted, fueled, cut):
		self.position.append(deepcopy(pos))
		self.sticky.append(deepcopy(sticky))
		self.fixed.append(deepcopy(fixed))
		self.cleaner.append(deepcopy(cleaner))
		self.actions.append(deepcopy(action))
		self.constraints.append(deepcopy(cons))
		self.metrics.append(deepcopy(metric))
		self.on.append(deepcopy(on))
		self.clean.append(deepcopy(clean))
		self.stick.append(deepcopy(stick))
		self.welded.append(deepcopy(welded))
		self.drilled.append(deepcopy(drilled))
		self.painted.append(deepcopy(painted))
		self.fueled.append(deepcopy(fueled))
		self.cut.append(deepcopy(cut))

	def addSymbolicAction(self, HLaction):
		self.symbolicActions.append(HLaction)

	def toString(self, delimiter='\n', subSymbolic=False, metrics=False):
		string = "World = " + self.world + "\nGoal = " + self.goal
		string += '\nSymbolic actions:\n'
		for action in self.symbolicActions:
			if str(action[0]) == 'E' or str(action[0]) == 'U':
				string = string + action + '\n'
				continue
			string = string + "\n".join(map(str, action)) + '\n'
		if not subSymbolic:
			return string
		string += 'States:\n'
		for i in range(len(self.position)):
			string = string + 'State ' + str(i) + ' ----------- ' + delimiter + \
				'Robot position - ' + str(self.position[i]) + delimiter + \
				'Sticky - ' + str(self.sticky[i]) + delimiter + \
				'Fixed - ' + str(self.fixed[i]) + delimiter + \
				'Cleaner? - ' + str(self.cleaner[i]) + delimiter + \
				'Objects-Cleaned? - ' + str(self.clean[i]) + delimiter + \
				'Stick with robot? - ' + str(self.stick[i]) + delimiter + \
				'Objects On - ' + str(self.on[i]) + delimiter + \
				'Objects welded - ' + str(self.welded[i]) + delimiter + \
				'Objects drilled - ' + str(self.drilled[i]) + delimiter + \
				'Objects painted - ' + str(self.painted[i]) + delimiter + \
				'Objects fueled - ' + str(self.fueled[i]) + delimiter + \
				'Objects cut - ' + str(self.cut[i]) + delimiter + \
				'Action - ' + str(self.actions[i]) + delimiter + \
				'Constraints - ' + str(self.constraints[i]) + delimiter
			if metrics:
				string = string + 'All metric - ' + str(self.metrics) + delimiter
		return string

	def readableSymbolicActions(self):
		string = 'Symbolic actions:\n\n'
		for action in self.symbolicActions:
			if str(action[0]) == 'E' or str(action[0]) == 'U':
				string = string + action + '\n'
				continue
			assert len(action) == 1
			dic = action[0]
			l = dic["args"]
			string = string + dic["name"] + "(" + str(l[0])
			for i in range(1, len(l)):
				string = string + ", " + str(l[i])
			string = string + ")\n"
		return string

	def getGraph(self, index=0, distance=False):
		world = 'home' if 'home' in self.world else 'factory' if 'factory' in self.world else 'outdoor'
		metrics = self.metrics[index]
		sceneobjects = list(metrics.keys())
		globalidlookup = globalIDLookup(sceneobjects, objects)
		nodes = []
		for obj in sceneobjects:
			if obj in skip: continue
			node = {}; objID = globalidlookup[obj]
			node['id'] = objID
			node['name'] = obj
			node['properties'] = objects[objID]['properties']
			if 'Movable' in node['properties'] and obj in self.fixed[index]: node['properties'].remove('Movable')
			states = []
			if obj in 'dumpster': states.append('Outside')
			else: states.append('Inside')
			if 'Switchable' in node['properties']:
				states.append('On') if obj in self.on[index] else states.append('Off')
			if 'Can_Open' in node['properties']:
				states.append('Close') if isInState(obj, allStates[world][obj]['close'], metrics[obj]) else states.append('Open')
			if 'Can_Lift' in node['properties']:
				states.append('Up') if isInState(obj, allStates[world][obj]['up'], metrics[obj]) else states.append('Down')
			if 'Stickable' in node['properties']:
				states.append('Sticky') if obj in self.sticky[index] else states.append('Non_Sticky')
			if 'Is_Dirty' in node['properties']:
				states.append('Dirty') if not obj in self.clean[index] else states.append('Clean')
			if 'Movable' in node['properties']:
				states.append('Grabbed') if grabbedObj(obj, self.constraints[index]) else states.append('Free')
			if 'Weldable' in node['properties']:
				states.append('Welded') if obj in self.welded[index] else states.append('Not_Welded')
			if 'Drillable' in node['properties']:
				states.append('Drilled') if obj in self.drilled[index] else states.append('Not_Drilled')
			if 'Drivable' in node['properties']:
				states.append('Driven') if obj in self.fixed[index] else states.append('Not_Driven')
			if 'Can_Fuel' in node['properties']:
				states.append('Fueled') if obj in self.fueled[index] else states.append('Not_Fueled')
			if 'Cuttable' in node['properties']:
				states.append('Cut') if obj in self.cut[index] else states.append('Not_Cut')
			if 'Can_Paint' in node['properties']:
				states.append('Painted') if obj in self.cut[index] else states.append('Not_Painted')
			node['states'] = states
			node['position'] = metrics[obj]
			node['size'] = objects[objID]['size']
			node['vector'] = objects[objID]['vector']
			nodes.append(node)
		edges = []
		for i in range(len(sceneobjects)):
			obj1 = sceneobjects[i]
			if obj1 in skip: continue
			for j in range(len(sceneobjects)):
				obj2 = sceneobjects[j]
				if obj2 in skip or i == j: continue
				obj1ID = globalidlookup[obj1]; obj2ID = globalidlookup[obj2]
				if checkNear(obj1, obj2, metrics):
					edges.append({'from': obj1ID, 'to': obj2ID, 'relation': 'Close'}) 
				if checkIn(obj1, obj2, objects[obj1ID], objects[obj2ID], metrics, self.constraints[index]):
					edges.append({'from': obj1ID, 'to': obj2ID, 'relation': 'Inside'}) 
				if checkOn(obj1, obj2, objects[obj1ID], objects[obj2ID], metrics, self.constraints[index]):
					edges.append({'from': obj1ID, 'to': obj2ID, 'relation': 'On'}) 
				if obj2 == 'walls' and 'Stickable' in objects[obj1ID]['properties'] and isInState(obj1, allStates[world][obj1]['stuck'], metrics[obj1]):
					edges.append({'from': obj1ID, 'to': obj2ID, 'relation': 'Stuck'}) 
				if distance:
					edges.append({'from': obj1ID, 'to': obj2ID, 'distance': getDirectedDist(obj1, obj2, metrics)})
		return {'graph_'+str(index): {'nodes': nodes, 'edges': edges}}

	def getTools(self, returnNoTool=False):
		goal_objects = getGoalObjects(self.world, self.goal)
		usedTools = []
		for action in self.actions:
			if 'Start' in action or 'Error' in action: continue
			for obj in action[1:]:
				if (not obj in goal_objects) and (not obj in usedTools) and obj in tools:
					usedTools.append(obj)
		if returnNoTool:	
			if (len(usedTools) == 0):
				usedTools.append("no-tool")
		return usedTools
