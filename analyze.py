from src.datapoint import *
import pickle
from os import listdir, remove, rename
import math
from scipy.spatial import distance
from statistics import mean 
from shutil import copyfile
import json
from src.extract_vectors import load_all_vectors
import numpy as np
import matplotlib.pyplot as plt
from statistics import pstdev
import seaborn as sns
import itertools
# from src.generalization import *

GOAL_LISTS = \
{'home': ["goal1-milk-fridge.json", "goal2-fruits-cupboard.json", "goal3-clean-dirt.json", "goal4-stick-paper.json", "goal5-cubes-box.json", "goal6-bottles-dumpster.json", "goal7-weight-paper.json", "goal8-light-off.json"],
'factory': ["goal1-crates-platform.json", "goal2-paper-wall.json", "goal3-board-wall.json", "goal4-generator-on.json", "goal5-assemble-parts.json", "goal6-tools-workbench.json", "goal7-clean-water.json", "goal8-clean-oil.json"]}

def printNumDatapoints(w='factory'):
	totalpoints = 0
	for goal in GOAL_LISTS[w]:
		print('Goal = ' + goal)
		goalpoints = 0
		for world in range(10):
			directory = './dataset/' + w + '/' + goal.split('.')[0] + '/world_' + w + str(world) + '/'
			try:
				numpoints = len(listdir(directory))
			except Exception as e:
				numpoints = 0
			goalpoints += numpoints
			print(numpoints, end = ' ')
		totalpoints += goalpoints
		print('\nTotal goal ' + goal + ' points = ' + str(goalpoints))
	print('Gross total points = ' + str(totalpoints))

def printDatapoint(filename):
	print(filename)
	f = open(filename + '.datapoint', 'rb')
	datapoint = pickle.load(f)
	print (datapoint.toString(subSymbolic=False))
	print(datapoint.getTools(returnNoTool=True), datapoint.totalTime()/10000)
	f.close()

def keepNewDatapoints(idx=1):
	for goal in GOAL_LIST:
		print('Goal = ' + goal)
		for world in range(10):
			directoryOld = './dataset1/home/' + goal.split('.')[0] + '/world_home' + str(world)
			directory = './dataset' + str(idx) + '/home/' + goal.split('.')[0] + '/world_home' + str(world)
			try: numpoints = len(listdir(directory))
			except Exception as e: numpoints = 0
			try: oldpoints = len(listdir(directoryOld))
			except Exception as e: oldpoints = 0
			for i in range(numpoints):
				if i < oldpoints: remove(directory + '/' + str(i) + '.datapoint')
				else: rename(directory + '/' + str(i) + '.datapoint', directory + '/' + str(i - oldpoints) + '.datapoint')

def printAllDatapoints():
	for goal in GOAL_LIST['factory']:
		print('Goal = ' + goal)
		for world in range(10):
			directory = './dataset/factory/' + goal.split('.')[0] + '/world_factory' + str(world)
			try:
				points = listdir(directory)
			except Exception as e:
				continue
			for point in points:
				printDatapoint(directory + '/' + point.split('.')[0])

def changeAllDatapoints():
	for goal in GOAL_LISTS['home']:
		if goal != "goal8-light-off.json": continue
		for world in [7]:
			directory = './dataset/home/' + goal.split('.')[0] + '/world_home' + str(world) + '/'
			for point in range(len(listdir(directory))):
				file = directory + str(point) + '.datapoint'
				f = open(file, 'rb')
				datapoint = pickle.load(f)
				f.close()
				if True:
					f = open(file, 'wb')
					for action in datapoint.symbolicActions:
						if action[0] != 'E' and action[0]['name'] == 'changeState' and action[0]['args'][1] != 'off':
							action[0]['args'][1] = 'off'
					pickle.dump(datapoint, f)
					f.flush()
					f.close()

def getTiming(goal='goal1-milk-fridge.json'):
	for world in range(10):
		directory = './dataset/home/' + goal.split('.')[0] + '/world_home' + str(world) + '/'
		times = []; actions = []; subactions = []
		for point in range(len(listdir(directory))):
			file = directory + str(point) + '.datapoint'
			f = open(file, 'rb')
			datapoint = pickle.load(f)
			times.append(totalTime(datapoint))
			actions.append(len(datapoint.symbolicActions))
			subactions.append(len(datapoint.actions))
		# print('World ' + str(world) + ' min = ' + str(min(times)) + ' avg = ' + str(mean(times)) + ' max = ' + str(max(times)))
		# print('World ' + str(world) + ' min = ' + str(min(actions)) + ' avg = ' + str(mean(actions)) + ' max = ' + str(max(actions)))
		print('World ' + str(world) + ' min = ' + str(min(subactions)) + ' avg = ' + str(mean(subactions)) + ' max = ' + str(max(subactions)))

def getInteractedObjs(datapoint):
	objs =  json.load(open("jsons/objects.json", "r"))["objects"]
	objs = [i['name'] for i in objs]
	os = []
	for action in datapoint.symbolicActions:
		if not (str(action[0]) == 'E' or str(action[0]) == 'U'):
			try: 
				for i in [0,1,2]: 
					if action[0]['args'][i] not in os and action[0]['args'][i] in objs: os.append(action[0]['args'][i])
			except: pass
	return os

def getAllData():
	domain = 'home'
	for goal in GOAL_LISTS[domain]:
		print(goal)
		times = []; actions = []; subactions = []; objs = []; tools = []
		for world in range(10):
			directory = './dataset/' + domain + '/' + goal.split('.')[0] + '/world_' + domain + str(world) + '/'
			for point in range(len(listdir(directory))):
				file = directory + str(point) + '.datapoint'
				f = open(file, 'rb')
				datapoint = pickle.load(f)
				times.append(datapoint.totalTime()/1000)
				actions.append(len(datapoint.symbolicActions))
				subactions.append(len(datapoint.actions))
				objs.append(len(getInteractedObjs(datapoint)))
				tools.append(len(datapoint.getTools()))
		print('Time       = ' + "{:.2f}".format(mean(times)) + ' +- ' + "{:.2f}".format(pstdev(times)/1000))
		print('Actions    = ' + "{:.2f}".format(mean(actions)) + ' +- ' + "{:.2f}".format(pstdev(actions)))
		print('SubActions = ' + "{:.2f}".format(mean(subactions)) + ' +- ' + "{:.2f}".format(pstdev(subactions)))
		print('Objects    = ' + "{:.2f}".format(mean(objs)) + ' +- ' + "{:.2f}".format(pstdev(objs)))
		print('Tools      = ' + "{:.2f}".format(mean(tools)) + ' +- ' + "{:.2f}".format(pstdev(tools)))

def combineDatasets(idx=1):
	for goal in GOAL_LIST:
		for world in range(10):
			directoryOld = './dataset/home/' + goal.split('.')[0] + '/world_home' + str(world)
			directory = './dataset'+ str(idx) + '/home/' + goal.split('.')[0] + '/world_home' + str(world)
			try: numpoints = len(listdir(directory))
			except Exception as e: numpoints = 0
			try: oldpoints = len(listdir(directoryOld))
			except Exception as e: oldpoints = 0
			for i in range(numpoints):
				copyfile(directory + '/' + str(i) + '.datapoint', directoryOld + '/' + str(i + oldpoints) + '.datapoint')

def printGraph(filename):
	print(filename)
	f = open(filename + '.datapoint', 'rb')
	datapoint = pickle.load(f); f.close()
	f = open("dataset/test/home/test2/0.graph", "w+")
	f.write(json.dumps(datapoint.getGraph(), indent=2))

def allActionTypes():
	actionTypes = []
	for goal in GOAL_LISTS['home']:
		for world in range(10):
			directory = './dataset/home/' + goal.split('.')[0] + '/world_home' + str(world) + '/'
			for point in range(len(listdir(directory))):
				file = directory + str(point) + '.datapoint'
				datapoint = pickle.load(open(file, 'rb'))
				for subAction in datapoint.symbolicActions:
					if len(subAction) == 1 and not [subAction[0]['name'], len(subAction[0]['args'])] in actionTypes:
						actionTypes.append([subAction[0]['name'], len(subAction[0]['args'])])
	print(actionTypes)

def allTools():
	tools = []
	for goal in GOAL_LISTS['factory']:
		for world in range(10):
			directory = './dataset/factory/' + goal.split('.')[0] + '/world_factory' + str(world) + '/'
			for point in range(len(listdir(directory))):
				file = directory + str(point) + '.datapoint'
				datapoint = pickle.load(open(file, 'rb'))
				for tool in datapoint.getTools():
					if tool not in tools:
						tools.append(tool)
	print(tools)

def checkActionTypes():
	actionTypes = []
	for goal in GOAL_LISTS['home']:
		print(goal)
		for world in range(10):
			directory = './dataset/home/' + goal.split('.')[0] + '/world_home' + str(world) + '/'
			for point in range(len(listdir(directory))):
				file = directory + str(point) + '.datapoint'
				datapoint = pickle.load(open(file, 'rb'))
				for action in datapoint.symbolicActions:
					if len(action) == 1:
						possible = getPossiblePredicates(action[0]['name'])
						for i in range(len(action[0]['args'])):
							if not action[0]['args'][i] in possible[i]:
								print(file)
								print(action[0]['name'], i, action[0]['args'][i])
							elif action[0]['name'] not in actionTypes: actionTypes.append(action[0]['name'])
	print(actionTypes)

def testData():
	# for i in range(1,9):	formTestData(i)
	formTestDataFactory(4)
	# for i in range(1,9): formTestDataFactory(i)

def printAllTimes():
	for goal in ["goal2-fruits-cupboard.json"]:
		print('Goal = ' + goal)
		for world in range(1):
			directory = './dataset/home/' + goal.split('.')[0] + '/world_home' + str(world)
			try:
				points = listdir(directory)
			except Exception as e:
				continue
			for point in points:
				printDatapoint(directory + '/' + point.split('.')[0])

def getInterestTools(domain, numTools):
	toolusage = {}
	TOOLS2 = ['stool', 'tray', 'tray2', 'big-tray', 'book', 'box', 'chair',\
			'stick', 'glue', 'tape', 'mop', 'sponge', 'vacuum'] if domain == 'home' else ['lift', \
			'stool', 'trolley', 'stick', 'ladder', 'glue', 'tape', 'drill', '3d_printer', \
			'screwdriver', 'brick', 'hammer', 'blow_dryer', 'box', 'wood_cutter', 'welder', \
			'spraypaint', 'toolbox', 'mop']
	for tool in TOOLS2:
		toolusage[tool] = 0
	for goal in GOAL_LISTS[domain]:
		print('Goal = ' + goal)
		for world in range(10):
			directory = './dataset/'+domain+'/' + goal.split('.')[0] + '/world_'+domain + str(world)
			try: points = listdir(directory)
			except Exception as e: continue
			for point in points:
				filename = directory + '/' + point.split('.')[0]
				f = open(filename + '.datapoint', 'rb')
				datapoint = pickle.load(f)
				ts = datapoint.getTools(returnNoTool=False)
				for t in ts: toolusage[t] += 1
				f.close()
	sortedtools = sorted(toolusage.items(), key=lambda kv: kv[1], reverse=True)
	print(sortedtools)
	interestTools = [a[0] for a in sortedtools][:numTools + (2)]
	if domain == 'factory': 
		interestTools.remove('spraypaint'); interestTools.remove('3d_printer');
	else:
		interestTools.remove('tray2'); interestTools.remove('sponge')
	return interestTools

def mapToolsGoals():
	numTools = 10; domain = 'home'
	interestTools = getInterestTools(domain, numTools)
	usemap = np.zeros((len(GOAL_LISTS[domain]), numTools))
	for goal in GOAL_LISTS[domain]:
		for world in range(10):
			directory = './dataset/'+domain+'/' + goal.split('.')[0] + '/world_'+domain + str(world)
			try: points = listdir(directory)
			except Exception as e: continue
			for point in points:
				filename = directory + '/' + point.split('.')[0]
				f = open(filename + '.datapoint', 'rb')
				datapoint = pickle.load(f)
				ts = datapoint.getTools(returnNoTool=False)
				for t in ts:
					if t in interestTools: usemap[GOAL_LISTS[domain].index(goal)][interestTools.index(t)] += 1
				f.close()
	print(interestTools)
	print(usemap)
	f, ax = plt.subplots(figsize=(4, 3))
	ax = sns.heatmap(usemap, cmap="Reds", yticklabels=list(range(1,9)), xticklabels=interestTools, linewidths=1)
	ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
	ax.set_ylabel(domain.capitalize() + ' Goal ID')
	plt.tight_layout()
	f.savefig('figures/'+domain+'_goal_tools.pdf')

def mapToolsWorlds():
	numTools = 10; domain = 'home'
	interestTools = getInterestTools(domain, numTools)
	usemap = np.zeros((10, numTools))
	for goal in GOAL_LISTS[domain]:
		for world in range(10):
			directory = './dataset/'+domain+'/' + goal.split('.')[0] + '/world_'+domain + str(world)
			try: points = listdir(directory)
			except Exception as e: continue
			for point in points:
				filename = directory + '/' + point.split('.')[0]
				f = open(filename + '.datapoint', 'rb')
				datapoint = pickle.load(f)
				ts = datapoint.getTools(returnNoTool=False)
				for t in ts:
					if t in interestTools: usemap[world][interestTools.index(t)] += 1
				f.close()
	print(interestTools)
	print(usemap)
	f, ax = plt.subplots(figsize=(4, 3))
	ax = sns.heatmap(usemap, cmap="Reds", yticklabels=list(range(10)), xticklabels=interestTools, linewidths=1)
	ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
	ax.set_ylabel(domain.capitalize() + ' Scene ID')
	plt.tight_layout()
	f.savefig('figures/'+domain+'_world_tools.pdf')

def getInteractedObjects(filename):
	f = open(filename + '.datapoint', 'rb')
	datapoint = pickle.load(f); f.close()
	os = []
	for action in datapoint.symbolicActions:
		if not (str(action[0]) == 'E' or str(action[0]) == 'U'):
			try: 
				for i in [1,2]: 
					if action[0]['args'][i] not in os: os.append(action[0]['args'][i])
			except: pass
	return os

def mapObjects():
	numObj = 5; domain = 'factory'
	objInteracted = {}
	for obj in json.load(open("jsons/objects.json", "r"))["objects"]:
		objInteracted[obj["name"]] = 0
	for goal in GOAL_LISTS[domain]:
		print('Goal = ' + goal)
		for world in range(10):
			directory = './dataset/'+domain+'/' + goal.split('.')[0] + '/world_'+domain + str(world)
			points = listdir(directory)
			for point in points:
				filename = directory + '/' + point.split('.')[0]
				for o in getInteractedObjects(filename): 
					try: objInteracted[o] += 1
					except: pass
	sortedObj = sorted(objInteracted.items(), key=lambda kv: kv[1], reverse=True)
	print(sortedObj)
	interestObj = [a[0] for a in sortedObj][:numObj]
	usemap = np.zeros((len(GOAL_LISTS[domain]), numObj))
	for goal in GOAL_LISTS[domain]:
		for world in range(10):
			directory = './dataset/'+domain+'/' + goal.split('.')[0] + '/world_'+domain + str(world)
			points = listdir(directory)
			for point in points:
				filename = directory + '/' + point.split('.')[0]
				for o in getInteractedObjects(filename): 
					if o in interestObj: usemap[GOAL_LISTS[domain].index(goal)][interestObj.index(o)] += 1
	print(interestObj)
	print(usemap)
	f, ax = plt.subplots(figsize=(4, 3))
	ax = sns.heatmap(usemap, cmap="Reds", yticklabels=[a.split('.')[0] for a in GOAL_LISTS[domain]], xticklabels=interestObj, linewidths=1)
	plt.tight_layout()
	f.savefig('figures/'+domain+'_objects.pdf')

def checkApprox():
	import approx
	for goal in GOAL_LISTS['home']:
		print('Goal = ' + goal)
		for world in range(10):
			directory = './dataset/home/' + goal.split('.')[0] + '/world_home' + str(world)
			try:
				points = listdir(directory)
			except Exception as e:
				continue
			for point in points:
				f = open(directory + '/' + point, 'rb')
				datapoint = pickle.load(f)
				args = approx.Args()
				args.world = 'jsons/home_worlds/world_home' + str(world) +'.json'
				args.goal = 'jsons/home_goals/' + goal
				plan = []
				# print("####### Goal", goal, "on world", world, "######")
				# print("####### Filename", point, " #######")
				for action in datapoint.symbolicActions:
					if str(action[0]) == 'E' or str(action[0]) == 'U':
						plan = []; break;
					else:
						plan.append(action[0])
				if plan == []: continue
				plan = {'actions': plan}
				approx.start(args)
				res = approx.execute(plan, args.goal, saveImg=False)
				assert res;
				f.close()

def get_all_possible_actions():
	actions = []
	for a in ["moveTo", "pick"]:
		for obj in all_objects:
			actions.append({'name': a, 'args':[obj]})
	actions.extend([{'name': i, 'args': ['stool']} for i in ["climbUp", "climbDown"]])
	actions.append({'name': 'clean', 'args': ['dirt']})
	for a in ["dropTo", "pushTo", "pickNplaceAonB"]:
		for obj in all_objects:
			for obj2 in all_objects:
				actions.append({'name': a, 'args':[obj, obj2]})
	for obj in ['glue', 'tape']:
		actions.append({'name': 'apply', 'args':[obj, 'paper']})
	actions.append({'name': 'stick', 'args': ['paper', 'walls']})
	for obj in all_objects_with_states:
		actions.extend([{'name': 'changeState', 'args':[obj, i]} for i in ['open', 'close']])
	actions.extend([{'name': 'changeState', 'args':['light', i]} for i in ['off']])
	return actions

def checkAllActions():
	import approx
	all_possible_actions = get_all_possible_actions()
	for goal in GOAL_LISTS['home']:
		print('Goal = ' + goal)
		for world in range(10):
			directory = './dataset/home/' + goal.split('.')[0] + '/world_home' + str(world)
			try:
				points = listdir(directory)
			except Exception as e:
				continue
			for point in points:
				f = open(directory + '/' + point, 'rb')
				datapoint = pickle.load(f)
				args = approx.Args()
				args.world = 'jsons/home_worlds/world_home' + str(world) +'.json'
				args.goal = 'jsons/home_goals/' + goal
				plan = []
				for action in datapoint.symbolicActions:
					if str(action[0]) == 'E' or str(action[0]) == 'U':
						plan = []; break;
					else:
						plan.append(action[0])
				if plan == []: continue
				for action in plan:
					if action not in all_possible_actions:
						print(action)

def checkPlan():
	import approx
	goal, world = 2, 0
	args = approx.Args()
	args.world = 'jsons/home_worlds/world_home' + str(world) +'.json'
	args.goal = 'jsons/home_goals/' + GOAL_LISTS['home'][goal]
	plan = [{'name': 'changeState', 'args': ['box', 'open']}, {'name': 'changeState', 'args': ['cupboard', 'open']}, {'name': 'pickNplaceAonB', 'args': ['apple', 'cupboard']}, {'name': 'pickNplaceAonB', 'args': ['orange', 'cupboard']}, {'name': 'pickNplaceAonB', 'args': ['banana', 'cupboard']}]
	plan = {'actions': plan}
	approx.start(args)
	approx.printAllValues()
	try:
		res = approx.execute(plan, args.goal, saveImg=False)
	except Exception as e:
		print(str(e))


# keepNewDatapoints(4)
# printAllDatapoints()
# printNumDatapoints(w='factory')
# changeAllDatapoints()
# combineDatasets(4)
# printGraph("dataset/factory/goal1-crates-platform/world_factory3/0")
# checkActionTypes()
# printGraph("dataset/home/goal1-milk-fridge/world_home4/0")
# testData()
# printAllTimes()
# allTools()
# mapToolsGoals()
# mapToolsWorlds()
# mapObjects()
# getAllData()
# checkApprox()
# checkPlan()
checkAllActions()