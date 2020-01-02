from husky_ur5 import *
from src.actions import *
from src.datapoint import Datapoint

def executeAction(inp):
    if execute(convertActionsFromFile(inp), './jsons/home_goals/goal1-milk-fridge.json'):
    	print("Goal Success!!!")
    else:
    	print("Goal Fail!!!")

# take input from user
inp = args.input
executeAction(inp)

datapoint = getDatapoint()
print(datapoint.toString(metrics=False))
saveDatapoint('test')
