import os
import time
import pdb
import pybullet as p
import pybullet_data
import math
import time
from src.initialise import *
from src.parser import *
from src.ur5 import *

object_file = "jsons/objects.json"
wings_file = "jsons/wings.json"

# Connect to Bullet using GUI mode
p.connect(p.GUI)

# Add input arguments
args = initParser()

# Initialize husky and ur5 model
husky, robotID, object_lookup, id_lookup, horizontal_list = initHuskyUR5(args.world, object_file)

# Fix ur5 to husky
cid = p.createConstraint(husky, -1, robotID, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], [0., 0., -.1],
                         [0, 0, 0, 1])

# Set small gravity
p.setGravity(0,0,-1)

# Initialize gripper joints and forces
controlJoints, joints = initGripper(robotID)
gotoWing = getUR5Controller(robotID)

# Position of the robot
x1,y1,o1 = 0,0,0
constraint = 0

# start simulation
book = id_lookup["book"]
try:
    while(True):
        keys = p.getKeyboardEvents()
        if 65297 in keys:
          x1 += math.cos(o1)*0.001
          y1 += math.sin(o1)*0.001
        if 65298 in keys:
          x1 -= math.cos(o1)*0.001
          y1 -= math.sin(o1)*0.001
        if 65295 in keys:
          o1 += 0.005
        if 65296 in keys:
          o1 -= 0.005
        gotoWing(robotID)
        q=p.getQuaternionFromEuler((0,0,o1))
        if p.getBasePositionAndOrientation(robotID)[0] != ((x1, y1, 0.220208), (q)):
            p.resetBasePositionAndOrientation(robotID, [x1, y1, 0.220208], q)
        if p.getBasePositionAndOrientation(husky)[0] != ((x1, y1, 0.0), (q)):
          p.resetBasePositionAndOrientation(husky, [x1, y1, 0], q)
        p.resetBasePositionAndOrientation(book, p.getBasePositionAndOrientation(book)[0], p.getQuaternionFromEuler((0,0,0)))
        p.stepSimulation()     
    p.disconnect()
except Exception as e: 
    print(e)
    p.disconnect()

