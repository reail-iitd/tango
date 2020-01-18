from src.datapoint import Datapoint
import pickle
from os import listdir

GOAL_LIST = ["goal1-milk-fridge.json", "goal2-fruits-cupboard.json", "goal3-clean-dirt.json", "goal4-stick-paper.json", "goal5-cubes-box.json", "goal6-bottles-dumpster.json", "goal7-weight-paper.json", "goal8-light-off.json"]

def printNumDatapoints():
	totalpoints = 0
	for goal in GOAL_LIST:
		print('Goal = ' + goal)
		goalpoints = 0
		for world in range(10):
			directory = './dataset/home/' + goal.split('.')[0] + '/world_home' + str(world)
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
	print(datapoint.toString(subSymbolic=False, metrics=False))
	f.close()

filename = './dataset/home/goal4-stick-paper/world_home0/'
for i in range(len(listdir(filename))):
	printDatapoint(filename+str(i))
# printDatapoint(filename)

# for goal in ["goal1-milk-fridge.json", "goal3-clean-dirt.json", "goal5-cubes-box.json", "goal8-light-off.json"]:
# 	for world in range(10):
# 		directory = './dataset/home/' + goal.split('.')[0] + '/world_home' + str(world) + '/'
# 		for point in range(len(listdir(directory))):
# 			file = directory + str(point) + '.datapoint'
# 			f = open(file, 'rb')
# 			datapoint = pickle.load(f)
# 			f.close()
# 			f = open(file, 'wb')
# 			datapoint.stick = [False] * len(datapoint.lighton)
# 			pickle.dump(datapoint, f)
# 			f.flush()
# 			f.close()

printNumDatapoints()