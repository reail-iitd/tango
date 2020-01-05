from src.datapoint import Datapoint
import pickle

def printDatapoint(filename):
	print(filename)
	f = open(filename + '.datapoint', 'rb')
	datapoint = pickle.load(f)
	print(datapoint.toString(metrics=False))
	f.close()

filename = './dataset/home/goal8-light-off/world_home0/7'
printDatapoint(filename)