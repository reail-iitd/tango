from husky_ur5 import *
from src.actions import *
from threading import Thread

def executeAction(inp):
    execute(convertActionsFromFile(inp))

while True:
    # take input from user
    inp = args.input
    # executeAction(inp)
    process = Thread(target=executeAction, args=[inp])
    process.start()
    # while images in folder, update images
    process.join()