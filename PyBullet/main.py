from husky_ur5 import *
from src.actions import *
from threading import Thread

def executeAction(inp):
    execute(convertActionsFromFile(inp))

# take input from user
inp = args.input
executeAction(inp)
