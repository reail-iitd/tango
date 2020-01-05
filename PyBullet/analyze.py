from src.datapoint import Datapoint
import pickle

def printDatapoint(filename):
	print(filename)
	f = open(filename + '.datapoint', 'rb')
	datapoint = pickle.load(f)
	print(datapoint.toString(metrics=False))
	f.close()

filename = './dataset/home/goal5-cubes-box/world_home1/4'
printDatapoint(filename)