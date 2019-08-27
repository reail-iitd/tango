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
from src.actions import *

object_file = "jsons/objects.json"
wings_file = "jsons/wings.json"
tolerance_file = "jsons/tolerance.json"

# Connect to Bullet using GUI mode
p.connect(p.GUI)

# Add input arguments
args = initParser()
speed = args.speed

# Initialize husky and ur5 model
( husky,
  robotID, 
  object_lookup, 
  id_lookup, 
  horizontal_list, 
  ground_list,
  tolerances, 
  cons_pos_lookup, 
  cons_link_lookup,
  ur5_dist) = initHuskyUR5(args.world, object_file)

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

# List of constraints with target object and constraint id
constraints = dict()

# List of low level actions
actions = convertActions(args.input)
print(actions)
action_index = 0
done = False
waiting = False
startTime = time.time()

# Start video recording
p.setRealTimeSimulation(0) 
logId = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "video.mp4") if args.logging else 0


# Start simulation
try:
    while(True):
        x1,y1,o1,keyboard = moveKeyboard(x1, y1, o1, [husky, robotID])
        moveUR5Keyboard(robotID, wings, gotoWing)
        keepHorizontal(horizontal_list)
        keepOnGround(ground_list)
        p.stepSimulation()  

        if action_index >= len(actions):
          if logId != -1 and args.logging:
            p.stopStateLogging(logId)
            p.disconnect()
            logId = -1
          continue

        if(actions[action_index][0] == "move"):
          target = actions[action_index][1]
          x1, y1, o1, done = move(x1, y1, o1, 
                                  [husky, robotID], 
                                  actions[action_index][1], 
                                  keyboard,
                                  speed)

        elif(actions[action_index][0] == "moveTo"):
          target = actions[action_index][1]
          x1, y1, o1, done = moveTo(x1, y1, o1, 
                                  [husky, robotID], 
                                  id_lookup[target], 
                                  tolerances[target], 
                                  keyboard,
                                  speed)

        elif(actions[action_index][0] == "changeWing"):
          if time.time()-startTime > 1:
            done = True
          pose = actions[action_index][1]
          gotoWing(robotID, wings[pose])

        elif(actions[action_index][0] == "constrain"):
          if time.time()-startTime > 1:
            done = True; waiting = False
          if not waiting and not done:
            cid = constrain(actions[action_index][1], 
                            actions[action_index][2], 
                            cons_link_lookup, 
                            cons_pos_lookup,
                            id_lookup,
                            constraints,
                            ur5_dist)
            constraints[actions[action_index][1]] = (actions[action_index][2], cid)
            waiting = True

        if done:
          startTime = time.time()
          action_index += 1
          done = False

    p.disconnect()
except Exception as e: 
    print(e)
    p.disconnect()

