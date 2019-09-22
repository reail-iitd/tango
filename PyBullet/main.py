from husky_ur5 import *
from src.actions import *
from threading import Thread

def executeAction():
    execute(convertActionsFromFile(args.input))

while True:
    # take input from user
    inp = args.input
    process = Thread(target=executeAction, args=[inp])
    process.start()
    # while images in folder, update images
    process.join()