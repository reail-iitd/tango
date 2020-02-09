from src.datapoint import Datapoint
import pickle
from os import listdir
import math
from scipy.spatial import distance
from statistics import mean 

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


def totalTime(dp):
	time = 0
	for i in range(len(dp.actions)):
		action = dp.actions[i][0]
		if action == 'S':
			continue
		dt = 0
		if action == 'moveTo' or action == 'moveToXY' or action == 'moveZ':
			x1 = dp.position[i][0]; y1 = dp.position[i][1]; o1 = dp.position[i][3]
			x2 = dp.metrics[i-1][dp.actions[i][1]][0][0]; y2 = dp.metrics[i-1][dp.actions[i][1]][0][1]
			dt = 100000 * abs(math.atan2(y2-y1,x2-x1) % (2*math.pi) - (o1%(2*math.pi)))
			dt += 2000 * abs(max(0.2, distance.euclidean((x1, y1, 0), (x2, y2, 0))) - 0.2)
			# print('rotate = ', str(100000 * abs(math.atan2(y2-y1,x2-x1) % (2*math.pi) - (o1%(2*math.pi)))))
			# print('move = ', str(2000 * abs(distance.euclidean((x1, y1, 0), (x2, y2, 0)))))
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
	# print(filename)
	f = open(filename + '.datapoint', 'rb')
	datapoint = pickle.load(f)
	# print(datapoint.toString(subSymbolic=False, metrics=False))
	print (datapoint.toString(subSymbolic=True))
	totalTime(datapoint)
	f.close()

filename = './dataset/home/goal8-light-off/world_home'
print ("Put the milk carton inside the fridge.\n\n")
for home_num in range(10):
	filename_home = filename + str(home_num) + "/"
	print ("World Number " + str(home_num) + "\n\n")
	try:
		for i in range(len(listdir(filename_home))):
			# print (filename_home)
			printDatapoint(filename_home+str(i))
	except FileNotFoundError:
		continue
printDatapoint(filename)

# for goal in GOAL_LIST:
# 	for world in range(10):
# 		directory = './dataset/home/' + goal.split('.')[0] + '/world_home' + str(world) + '/'
# 		for point in range(len(listdir(directory))):
# 			file = directory + str(point) + '.datapoint'
# 			f = open(file, 'rb')
# 			datapoint = pickle.load(f)
# 			f.close()
# 			f = open(file, 'wb')
# 			datapoint.on = []
# 			for i in range(len(datapoint.lighton)):
# 				a = ['light'] if datapoint.lighton[i] else []
# 				datapoint.on.append(a)
# 			pickle.dump(datapoint, f)
# 			f.flush()
# 			f.close()

printNumDatapoints()

# goal = 'goal1-milk-fridge.json'
# for world in range(10):
# 	directory = './dataset/home/' + goal.split('.')[0] + '/world_home' + str(world) + '/'
# 	times = []; actions = []; subactions = []
# 	for point in range(len(listdir(directory))):
# 		file = directory + str(point) + '.datapoint'
# 		f = open(file, 'rb')
# 		datapoint = pickle.load(f)
# 		times.append(totalTime(datapoint))
# 		actions.append(len(datapoint.symbolicActions))
# 		subactions.append(len(datapoint.actions))
# 	# print('World ' + str(world) + ' min = ' + str(min(times)) + ' avg = ' + str(mean(times)) + ' max = ' + str(max(times)))
# 	# print('World ' + str(world) + ' min = ' + str(min(actions)) + ' avg = ' + str(mean(actions)) + ' max = ' + str(max(actions)))
# 	print('World ' + str(world) + ' min = ' + str(min(subactions)) + ' avg = ' + str(mean(subactions)) + ' max = ' + str(max(subactions)))
