from husky_ur5 import *
from src.actions import *
from src.datapoint import Datapoint
from src.parser import *

args = initParser()

def executeAction(inp):
    start(args)
    if execute(convertActionsFromFile(inp), args.goal):
    	print("Goal Success!!!")
    else:
    	print("Goal Fail!!!")

# take input from user
inp = args.input
executeAction(inp)

datapoint = getDatapoint()
print(datapoint.toString(metrics=False))
saveDatapoint('test')
