from src.datapoint import *
import pickle
from os import listdir, remove, rename
import math
from scipy.spatial import distance
from statistics import mean 
from shutil import copyfile
import json

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


def totalTime(dp):
	time = 0
	for i in range(len(dp.actions)):
		action = dp.actions[i][0]
		if action == 'S':
			continue
		dt = 0
		if action == 'moveTo' or action == 'moveToXY' or action == 'moveZ':
			x1 = dp.position[i][0]; y1 = dp.position[i][1]; o1 = dp.position[i][3]
			if 'list' in str(type(dp.actions[i][1])):
				x2 = dp.actions[i][1][0]; y2 = dp.actions[i][1][1]
			else:
				x2 = dp.metrics[i-1][dp.actions[i][1]][0][0]; y2 = dp.metrics[i-1][dp.actions[i][1]][0][1]
			robot, dest = o1%(2*math.pi), math.atan2((y2-y1),(x2-x1))%(2*math.pi)
			left = (robot - dest)%(2*math.pi); right = (dest - robot)%(2*math.pi)
			dt = 100000 * abs(min(left, right)) # time for rotate
			dt += 2000 * abs(max(0.2, distance.euclidean((x1, y1, 0), (x2, y2, 0))) - 0.2) # time for move
		elif action == 'move':
			x1 = dp.position[i][0]; y1 = dp.position[i][1]; o1 = dp.position[i][3]
			x2 = -2; y2 = 3
			dt = 100000 * abs(math.atan2(y2-y1,x2-x1) % (2*math.pi) - (o1%(2*math.pi)))
			dt += 2000 * abs(max(0.2, distance.euclidean((x1, y1, 0), (x2, y2, 0))) - 0.2)
		elif action == 'constrain' or action == 'removeConstraint' or action == 'changeWing':
			dt = 1000
		elif action == 'climbUp' or action == 'climbDown' or action == 'changeState':
			dt = 1200
		# print("Action = ", action, ", Time = ", dt)
		time += dt
	return time

def printDatapoint(filename):
	print(filename)
	f = open(filename + '.datapoint', 'rb')
	datapoint = pickle.load(f)
	print (datapoint.toString(subSymbolic=True))
	print(totalTime(datapoint))
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


# keepNewDatapoints(4)
# printAllDatapoints()
# printNumDatapoints()
# changeAllDatapoints()
# combineDatasets(4)
# printGraph("dataset/factory/goal1-crates-platform/world_factory3/0")
# checkActionTypes()
printGraph("dataset/home/goal1-milk-fridge/world_home4/0")

