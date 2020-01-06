from src.datapoint import Datapoint
import pickle
from os import listdir

def printDatapoint(filename):
	print(filename)
	f = open(filename + '.datapoint', 'rb')
	datapoint = pickle.load(f)
	print(datapoint.toString(metrics=False))
	f.close()

filename = './dataset1/home/goal3-clean-dirt/world_home0/0'
printDatapoint(filename)

# for goal in ['goal5-cubes-box', 'goal8-light-off']:
# 	for world in range(10):
# 		directory = './dataset1/home/' + goal + '/world_home' + str(world) + '/'
# 		for point in range(len(listdir(directory))):
# 			file = directory + str(point) + '.datapoint'
# 			f = open(file, 'rb')
# 			datapoint = pickle.load(f)
# 			f.close()
# 			f = open(file, 'wb')
# 			datapoint.dirtClean = [False] * len(datapoint.lighton)
# 			pickle.dump(datapoint, f)
# 			f.flush()
# 			f.close()