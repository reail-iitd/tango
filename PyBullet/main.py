from husky_ur5 import *
from src.actions import *
from src.datapoint import Datapoint

def executeAction(inp):
    execute(convertActionsFromFile(inp))

# take input from user
inp = args.input
executeAction(inp)

datapoint = getDatapoint()
print(datapoint.toString(metrics=False))
saveDatapoint('test')
