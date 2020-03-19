from src.datapoint import *
import pickle
from os import listdir, remove, rename
import math
from scipy.spatial import distance
from statistics import mean 
from shutil import copyfile
import json
from extract_vectors import load_all_vectors
from src.generalization import *

GOAL_LISTS = \
{'home': ["goal1-milk-fridge.json", "goal2-fruits-cupboard.json", "goal3-clean-dirt.json", "goal4-stick-paper.json", "goal5-cubes-box.json", "goal6-bottles-dumpster.json", "goal7-weight-paper.json", "goal8-light-off.json"],
'factory': ["goal1-crates-platform.json", "goal2-paper-wall.json", "goal3-board-wall.json", "goal4-generator-on.json", "goal5-assemble-parts.json", "goal6-tools-workbench.json", "goal7-clean-water.json", "goal8-clean-oil.json"]}

def printNumDatapoints(w='factory'):
	totalpoints = 0
	for goal in GOAL_LISTS[w]:
		print('Goal = ' + goal)
		goalpoints = 0
		for world in range(10):
			directory = './dataset/factory/' + goal.split('.')[0] + '/world_' + w + str(world) + '/'
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
	for goal in GOAL_LIST:
		print('Goal = ' + goal)
		for world in range(10):
			directory = './dataset/home/' + goal.split('.')[0] + '/world_home' + str(world)
			try:
				points = listdir(directory)
			except Exception as e:
				continue
			for point in points:
				printDatapoint(directory + '/' + point.split('.')[0])

def changeAllDatapoints():
	for goal in GOAL_LIST:
		for world in range(10):
			directory = './dataset/home/' + goal.split('.')[0] + '/world_home' + str(world) + '/'
			for point in range(len(listdir(directory))):
				file = directory + str(point) + '.datapoint'
				f = open(file, 'rb')
				datapoint = pickle.load(f)
				f.close()
				if True:
					f = open(file, 'wb')
					datapoint.world = 'world_home' + str(world)
					datapoint.goal = goal.split('.')[0]
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

def checkActionTypes():
	actionTypes = []
	for goal in GOAL_LISTS['home']:
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

def testData():
	# for i in range(1,9):
	# 	formTestData(i)
	formTestData(9)

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

# keepNewDatapoints(4)
# printAllDatapoints()
# printNumDatapoints()
# changeAllDatapoints()
# combineDatasets(4)
# printGraph("dataset/factory/goal1-crates-platform/world_factory3/0")
# checkActionTypes()
# printGraph("dataset/home/goal1-milk-fridge/world_home4/0")
testData()
# printAllTimes()
