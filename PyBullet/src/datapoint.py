from copy import deepcopy
from src.utils import *
import json

tools = ['stool', 'tray', 'tray2', 'lift', 'ramp', 'big-tray', 'book', 'box', 'chair',\
		'stick', 'glue', 'tape', 'mop', 'sponge', 'vacuum', 'drill', 'screwdriver',\
		'hammer', 'ladder', 'trolley', 'brick', 'blow_dryer']

skip = ['ur5', 'cupboard_back', 'fridge_back']

objects = None
with open('jsons/objects.json', 'r') as handle:
    objects = json.load(handle)['objects']

allStates = None
with open('jsons/states.json', 'r') as handle:
    allStates = json.load(handle)

class Datapoint:
	def __init__(self):
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
		# Dirt Cleaned
		self.dirtClean = []
		# Stick with object
		self.stick = []
		# Time
		self.time = 0

	def addPoint(self, pos, sticky, fixed, cleaner, action, cons, metric, on, dirtClean, stick):
		self.position.append(deepcopy(pos))
		self.sticky.append(deepcopy(sticky))
		self.fixed.append(deepcopy(fixed))
		self.cleaner.append(deepcopy(cleaner))
		self.actions.append(deepcopy(action))
		self.constraints.append(deepcopy(cons))
		self.metrics.append(deepcopy(metric))
		self.on.append(deepcopy(on))
		self.dirtClean.append(deepcopy(dirtClean))
		self.stick.append(deepcopy(stick))

	def addSymbolicAction(self, HLaction):
		self.symbolicActions.append(HLaction)

	def toString(self, delimiter='\n', subSymbolic=False, metrics=False):
		string = 'Symbolic actions:\n'
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
				'Dirt-Cleaned? - ' + str(self.dirtClean[i]) + delimiter + \
				'Stick with robot? - ' + str(self.stick[i]) + delimiter + \
				'Objects On - ' + str(self.on[i]) + delimiter + \
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

	def getGraph(self, world='home', index=0, distance = False):
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
			if 'Stickable' in node['properties']:
				states.append('Sticky') if obj in self.sticky[index] else states.append('Non_Sticky')
			if 'Is_Dirty' in node['properties']:
				states.append('Dirty') if not self.dirtClean[index] else states.append('Clean')
			if 'Movable' in node['properties']:
				states.append('Grabbed') if grabbedObj(obj, self.constraints[index]) else states.append('Free')
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

	def getTools(self, goal_objects):
		usedTools = []
		for action in self.actions:
			if 'Start' in action or 'Error' in action: continue
			for obj in action[1:]:
				if (not obj in goal_objects) and (not obj in usedTools) and obj in tools:
					usedTools.append(obj)
		return usedTools
