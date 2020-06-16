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

import warnings
warnings.simplefilter("ignore")

plt.style.use(['science'])
plt.rcParams["text.usetex"] = True

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

def allActionTypes(d):
	actionTypes = []
	for goal in GOAL_LISTS[d]:
		for world in range(10):
			directory = './dataset/' + d + '/' + goal.split('.')[0] + '/world_' + d + str(world) + '/'
			for point in range(len(listdir(directory))):
				file = directory + str(point) + '.datapoint'
				datapoint = pickle.load(open(file, 'rb'))
				for subAction in datapoint.symbolicActions:
					if len(subAction) == 1 and not [subAction[0]['name'], len(subAction[0]['args'])] in actionTypes:
						actionTypes.append([subAction[0]['name'], len(subAction[0]['args'])])
	print(actionTypes)
	return actionTypes

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
	for goal in GOAL_LISTS[domain]:
		print(goal)
		for world in range(10):
			directory = './dataset/'+domain+'/' + goal.split('.')[0] + '/world_' + domain + str(world) + '/'
			for point in range(len(listdir(directory))):
				file = directory + str(point) + '.datapoint'
				datapoint = pickle.load(open(file, 'rb'))
				for action in datapoint.symbolicActions:
					if str(action[0]) == 'E' or str(action[0]) == 'U': break
					if len(action) == 1:
						possible = getPossiblePredicates(action[0]['name'])
						for i in range(len(action[0]['args'])):
							if not action[0]['args'][i] in possible[i]:
								print(file)
								print(action[0], i, action[0]['args'][i])
							elif action[0]['name'] not in actionTypes: actionTypes.append(action[0]['name'])
	print(actionTypes)

def testData():
	# for i in range(1,10):	formTestData(i)
	formTestDataFactory(7)
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
				for i in [0,1]: 
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

def allObjects():
	objInteracted = []
	allobjs = [o['name'] for o in json.load(open("jsons/objects.json", "r"))["objects"]]
	for goal in GOAL_LISTS[domain]:
		print('Goal = ' + goal)
		for world in range(10):
			directory = './dataset/'+domain+'/' + goal.split('.')[0] + '/world_'+domain + str(world)
			points = listdir(directory)
			for point in points:
				filename = directory + '/' + point.split('.')[0]
				for o in getInteractedObjects(filename): 
					if o not in objInteracted and o in allobjs: objInteracted.append(o)
	objInteracted.sort()
	print(objInteracted)

def checkApprox(d):
	import approx
	for goal in GOAL_LISTS[d]:
		print('Goal = ' + goal)
		for world in range(10):
			directory = './dataset/' + d + '/' + goal.split('.')[0] + '/world_' + d + str(world)
			try:
				points = listdir(directory)
			except Exception as e:
				continue
			for point in points:
				f = open(directory + '/' + point, 'rb')
				datapoint = pickle.load(f)
				args = approx.Args()
				args.world = 'jsons/' + d + '_worlds/world_' + d + str(world) +'.json'
				args.goal = 'jsons/' + d + '_goals/' + goal
				plan = []
				# print("####### Goal", goal, "on world", world, "######")
				# print("####### Filename", point, " #######")
				for action in datapoint.symbolicActions:
					if str(action[0]) == 'E' or str(action[0]) == 'U':
						plan = []; break;
					else:
						plan.append(action[0])
				if plan == []: continue
				# print(plan)
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

def listSum(a, b):
    c = []
    for i in range(len(a)):
        c.append(round(a[i]+b[i], 4))
    return c

def accuracyWithTime():
	# a = eval('[[5, 0, 0], [2, 0, 0], [7, 0, 0], [16, 0, 1], [13, 0, 1], [4, 0, 1], [3, 0, 3], [5, 0, 0], [2, 0, 0], [2, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]')
	# b = eval('[[0, 0, 0], [0, 0, 0], [9, 0, 0], [3, 0, 1], [0, 0, 0], [0, 0, 0], [4, 0, 4], [8, 0, 2], [4, 0, 4], [3, 0, 0], [2, 0, 5], [5, 0, 1], [1, 0, 1], [5, 0, 1], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 1], [2, 0, 1], [0, 0, 0], [5, 0, 1]]')
	# res = []
	# for i in range(len(a)):
	# 	d = []
	# 	for j in range(3):
	# 		d.append(a[i][j] + b[i][j])
	# 	res.append(d)
	## test both
	res = eval('[[5, 0, 0], [2, 0, 0], [16, 0, 0], [19, 0, 2], [13, 0, 1], [4, 0, 1], [7, 0, 7], [13, 0, 2], [6, 0, 5], [5, 0, 6], [2, 0, 5], [5, 0, 3], [1, 0, 1], [3, 0, 4], [1, 0, 1], [1, 0, 2], [1, 0, 3], [1, 0, 2], [2, 0, 5], [0, 0, 0], [3, 0, 5]]')
	## train home
	# res = eval('[[47, 0, 12], [51, 0, 5], [84, 4, 7], [112, 2, 10], [83, 1, 21], [46, 0, 17], [37, 0, 24], [34, 0, 10], [15, 0, 47], [19, 0, 20], [5, 0, 10], [3, 0, 9], [0, 0, 0], [4, 0, 10]]')
	## train factory
	# res = eval('[[0, 0, 0], [0, 0, 0], [42, 0, 5], [31, 0, 7], [18, 0, 10], [66, 0, 14], [47, 0, 42], [24, 0, 32], [24, 0, 38], [26, 0, 37], [20, 0, 28], [22, 0, 6], [8, 0, 13], [11, 1, 18], [10, 0, 6], [6, 0, 1], [6, 0, 7], [7, 0, 1], [10, 1, 4], [2, 0, 1], [1, 0, 1], [1, 3, 3], [3, 0, 3], [0, 0, 3], [0, 7, 7], [2, 4, 0], [0, 0, 1], [0, 0, 0], [0, 2, 3], [1, 4, 6]]')
	c, i, e  = [], [], []
	for j in range(len(res)):
		if sum(res[j]) != 0:
			# c.append(100*res[j][0]/sum(res[j])); i.append(100*res[j][1]/sum(res[j])); e.append(100*res[j][2]/sum(res[j]))
			c.append(res[j][0]); i.append(res[j][1]); e.append(res[j][2])
		else:
			c.append(0); i.append(0); e.append(0)
	fig = plt.figure(figsize=(3,2.5))
	plt.xticks(range(0, len(c), 4), range(1, len(c)+1, 4))
	# plt.plot(range(len(c)), c, '.-', label='Home+Factory')
	plt.bar(range(len(c)), c, label='Successful', edgecolor='k')
	# plt.bar(range(len(c)), i, bottom=c, label='Incorrect', edgecolor='k')
	plt.bar(range(len(c)), e, bottom=listSum(c,i), label='Unsuccessful', edgecolor='k')
	plt.legend(ncol=1)
	plt.ylabel('Plan Execution Performance (\%)')
	plt.xlabel('Plan length')
	plt.tight_layout()
	plt.savefig('test.pdf')

def planLen():
	## test home
	# human = eval('[3.375, 3.375, 3.625, 5.75, 3.7142857142857144, 7.75, 0.2857142857142857, 3.090909090909091]')
	# model = eval('[3.0, 4.285714285714286, 4.125, 8.0, 14.0, 10.0, 1.0, 1.0]')
	# humandev = eval('[1.7677669529663689, 0.5175491695067657, 1.0606601717798212, 0.4629100498862757, 0.4879500364742666, 0.8864052604279183, 0.1879500364742666, 0.8312094145936335]')
	# modeldev = eval('[0.0, 0.7559289460184544, 0.3535533905932738, 0.0, 0.0, 0.0, 0.0, 0.0]')
	## test factory
	human = eval('[11.363636363636363, 10.4, 11.8, 8.0, 19.333333333333332, 7.0, 5.333333333333333, 2.0]')
	model = eval('[13.11111111111111, 10.142857142857142, 9.25, 9.829, 26.0, 8.0, 4.0, 3.888888888888889]')
	humandev = eval('[3.8800187440971796, 1.429840705968481, 2.780887148615228, 1.8856180831641267, 1.0, 0.0, 2.5, 0.0]')
	modeldev = eval('[1.536590742882148, 1.4638501094227998, 0.7071067811865476, 0, 1.9148542155126762, 0.0, 0.0, 0.33333333333333337]')
	fig = plt.figure(figsize=(3,2.5))
	plt.xticks(np.arange(len(human)), np.arange(1, len(human)+1, 1))
	plt.bar(np.arange(len(human))-0.2, human, yerr=humandev, capsize=1, ecolor='black', width=0.4, label='Human', edgecolor='k')
	plt.bar(np.arange(len(human))+0.2, model, yerr=modeldev, capsize=1, ecolor='black', width=0.4, label='Model', edgecolor='k')
	# plt.legend(loc=9, ncol=3,bbox_to_anchor=(0.5,1.2))
	plt.ylabel('Average Plan Length')
	plt.xlabel('Goal ID')
	plt.tight_layout()
	plt.savefig('test_factory.pdf')

def datasetObjInt():
	res = []; topX = 10
	allobjs = [o['name'] for o in json.load(open("jsons/objects.json", "r"))["objects"]]
	for domain in ['home', 'factory']:
		timesInteracted = {};
		for i in allobjs: timesInteracted[i] = 0
		for goal in GOAL_LISTS[domain]:
			for world in range(10):
				directory = './dataset/'+domain+'/' + goal.split('.')[0] + '/world_'+ domain + str(world)
				points = listdir(directory)
				for point in points:
					objs = getInteractedObjects(directory + '/' + str(point).split('.')[0])
					for o in objs: 
						if o in allobjs: timesInteracted[o] += 1
		sort_orders = sorted(timesInteracted.items(), key=lambda x: x[1], reverse=True)
		res.append(([i[0].replace('_',' ') for i in sort_orders[:topX]],[i[1] for i in sort_orders[:topX]]))
	fig = plt.figure(figsize=(3,2.5))
	print(res[1])
	plt.xticks(np.arange(0, len(res[1][0])), res[1][0], rotation=90)
	plt.bar(np.arange(len(res[1][1])), res[1][1], label='Factory', edgecolor='k')
	plt.ylabel('Number of interactions')
	plt.legend()
	plt.tight_layout()
	plt.savefig('num_int.pdf')

def datasetObjIntAll():
	res = []; topX = 10
	allobjs = [o['name'] for o in json.load(open("jsons/objects.json", "r"))["objects"]]
	for domain in ['home', 'factory']:
		timesInteracted = {};
		for i in allobjs: timesInteracted[i] = 0
		timesInteracted['wall'] = 0
		for goal in GOAL_LISTS[domain]:
			for world in range(10):
				directory = './dataset/'+domain+'/' + goal.split('.')[0] + '/world_'+ domain + str(world)
				points = listdir(directory)
				for point in points:
					objs = getInteractedObjects(directory + '/' + str(point).split('.')[0])
					for o in objs: 
						if o in allobjs: timesInteracted[o.replace('walls', 'wall').replace('_warehouse','')] += 1
		res.append(timesInteracted)
	allTimes = {}
	for o in allobjs+['wall']: 
		allTimes[o] = (res[0][o] if o in res[0] else 0) + (res[1][o] if o in res[1] else 0)
	sort_orders = sorted(allTimes.items(), key=lambda x: x[1], reverse=True)
	finalobjs = [i[0] for i in sort_orders[:topX]]
	a = [res[0][o] if o in res[0] else 0 for o in finalobjs]; b = [res[1][o] if o in res[1] else 0 for o in finalobjs]
	print(a); print(b)
	fig = plt.figure(figsize=(3,2.5))
	plt.xticks(np.arange(0, len(finalobjs)), [i.replace('_warehouse','') for i in finalobjs], rotation=90)
	plt.bar(np.arange(len(a)), a, label='Home', color='C3', edgecolor='k')
	plt.bar(np.arange(len(b)), b, label='Factory', color='C4', bottom=a, edgecolor='k')
	plt.ylabel('Number of interactions')
	plt.legend()
	plt.tight_layout()
	plt.savefig('num_int.pdf')

def datasetPlanLen():
	res = []
	for domain in ['home', 'factory']:
		planLen = []; maxplanlen = 14 if domain == 'home' else 32; 
		for i in range(maxplanlen): planLen.append(0)
		for goal in GOAL_LISTS[domain]:
			for world in range(10):
				directory = './dataset/'+domain+'/' + goal.split('.')[0] + '/world_'+ domain + str(world)
				points = listdir(directory)
				for point in points:
					file = directory + '/' + str(point)
					datapoint = pickle.load(open(file, 'rb'))
					planLen[len(datapoint.symbolicActions)-1] += 1
		res.append(planLen)
	fig = plt.figure(figsize=(3,2.5))
	plt.xticks(np.arange(0, len(res[1]), 4), np.arange(1, len(res[1])+1, 4))
	plt.plot(np.arange(len(res[0])), res[0], '.-', color='C3', label='Home')
	plt.plot(np.arange(len(res[1])), res[1], '.-', color='C4', label='Factory')
	plt.ylabel('Number of plans')
	plt.xlabel('Plan Length')
	plt.legend()
	plt.tight_layout()
	plt.savefig('planlen.pdf')

def datasetActionsAll():
	res = []; topX = 10; x = allActionTypes('home'); y = allActionTypes('factory')
	allActions = list(set().union([i[0] for i in x],[i[0] for i in y]))
	for domain in ['home', 'factory']:
		timesInteracted = {};
		for i in allActions: timesInteracted[i] = 0
		for goal in GOAL_LISTS[domain]:
			for world in range(10):
				directory = './dataset/'+domain+'/' + goal.split('.')[0] + '/world_'+ domain + str(world)
				points = listdir(directory)
				for point in points:
					file = directory + '/' + str(point)
					datapoint = pickle.load(open(file, 'rb'))
					for subAction in datapoint.symbolicActions:
						if len(subAction) == 1 and subAction[0]['name'] in allActions:
							timesInteracted[subAction[0]['name']] += 1
		res.append(timesInteracted)
	allTimes = {}
	for o in allActions: 
		allTimes[o] = (res[0][o] if o in res[0] else 0) + (res[1][o] if o in res[1] else 0)
	sort_orders = sorted(allTimes.items(), key=lambda x: x[1], reverse=True)
	finalobjs = [i[0] for i in sort_orders[:topX]]
	a = [res[0][o] if o in res[0] else 0 for o in finalobjs]; b = [res[1][o] if o in res[1] else 0 for o in finalobjs]
	print(a); print(b)
	fig = plt.figure(figsize=(3,2.5))
	plt.xticks(np.arange(0, len(finalobjs)), [i.replace('pickNplaceAonB', 'pick and place') for i in finalobjs], rotation=90)
	plt.bar(np.arange(len(a)), a, label='Home', color='C3', edgecolor='k')
	plt.bar(np.arange(len(b)), b, label='Factory', color='C4', bottom=a, edgecolor='k')
	plt.ylabel('Frequency of action')
	plt.legend()
	plt.tight_layout()
	plt.savefig('num_actions.pdf')

# keepNewDatapoints(4)
# printAllDatapoints()
# printNumDatapoints(w='factory')
# changeAllDatapoints()
# combineDatasets(4)
# printGraph("dataset/factory/goal1-crates-platform/world_factory3/0")
# printGraph("dataset/home/goal1-milk-fridge/world_home4/0")
# testData()
# printAllTimes()
# allTools()
# mapToolsGoals()
# mapToolsWorlds()
# mapObjects()
# getAllData()
# allObjects()
# allActionTypes('factory')
# checkActionTypes()
# printDatapoint('dataset/factory/goal4-generator-on/world_factory9/3')
# checkApprox(domain)
# checkPlan()
# checkAllActions()
# accuracyWithTime()
# planLen()
# datasetPlanLen()
# datasetObjInt()
datasetObjIntAll()
# datasetActionsAll()