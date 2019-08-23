import os
import time
import pdb
import pybullet as p
import time
from src.initialise import *
from src.parser import *
from src.ur5 import *
from src.utils import *
from src.basic_actions import *

object_file = "jsons/objects.json"
wings_file = "jsons/wings.json"
tolerance_file = "jsons/tolerance.json"

# Connect to Bullet using GUI mode
p.connect(p.GUI)

# Add input arguments
args = initParser()

# Initialize husky and ur5 model
husky, robotID, object_lookup, id_lookup, horizontal_list, tolerances = initHuskyUR5(args.world, object_file)

# Initialize dictionary of wing positions
wings = initWingPos(wings_file)

# Fix ur5 to husky
cid = p.createConstraint(husky, -1, robotID, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], [0., 0., -.1],
                         [0, 0, 0, 1])

# Set small gravity
p.setGravity(0,0,-1)

# Initialize gripper joints and forces
controlJoints, joints = initGripper(robotID)
gotoWing = getUR5Controller(robotID)
gotoWing(robotID, wings["home"])

# Position of the robot
x1,y1,o1 = 0,0,0
constraint = 0

# List of low level actions
actions = [["moveTo", "big-tray"],
           ["changeWing", "up"],
           ["moveTo", "r2d2"],
           ["moveTo", "bottle_blue"],
           ["moveTo", "chair"],
           ["moveTo", "apple"]]
action_index = 0
done = False
startTime = time.time()

# Start simulation
try:
    while(True):
        x1,y1,o1,moved = moveKeyboard(x1, y1, o1, [husky, robotID])
        moveUR5Keyboard(robotID, wings, gotoWing)

        if(actions[action_index][0] == "moveTo"):
          target = actions[action_index][1]
          x1, y1, o1, done = moveTo(x1, y1, o1, 
                                  [husky, robotID], 
                                  id_lookup[target], 
                                  tolerances[target], 
                                  moved)
        elif(actions[action_index][0] == "changeWing"):
          if time.time()-startTime > 1:
            done = True
          pose = actions[action_index][1]
          gotoWing(robotID, wings[pose])

        if done:
          startTime = time.time()
          action_index += 1
          done = False

        keepHorizontal(horizontal_list)
        p.stepSimulation()     
    p.disconnect()
except Exception as e: 
    print(e)
    p.disconnect()

