import os
import time
import pdb
import pybullet as p
import time
import os, shutil
from src.initialise import *
from src.parser import *
from src.ur5 import *
from src.utils import *
from src.basic_actions import *
from src.actions import *

object_file = "jsons/objects.json"
wings_file = "jsons/wings.json"
tolerance_file = "jsons/tolerance.json"
goal_file = "jsons/goal.json"

# Enclosures
enclosures = ['fridge', 'cupboard']

# Sticky objects
sticky = []

# Connect to Bullet using GUI mode
light = p.connect(p.GUI)

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
  fixed_orientation,
  tolerances, 
  cons_pos_lookup, 
  cons_link_lookup,
  ur5_dist,
  states) = initHuskyUR5(args.world, object_file)

# Initialize dictionary of wing positions
wings = initWingPos(wings_file)

# Fix ur5 to husky
cid = p.createConstraint(husky, -1, robotID, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, ], [0., 0., -.2],
                         [0, 0, 0, 1])

# Set small gravity
p.setGravity(0,0,-10)

# Initialize gripper joints and forces
controlJoints, joints = initGripper(robotID)
gotoWing = getUR5Controller(robotID)
gotoWing(robotID, wings["home"])

# Position of the robot
x1, y1, o1 = 0, 0, 0
constraint = 0

# List of constraints with target object and constraint id
constraints = dict()

# List of low level actions
actions = convertActions(args.input)
print(actions)
action_index = 0
done = False
waiting = False
startTime = current_milli_time()
lastTime = startTime

# Init camera
imageCount = 0
yaw = 50
ims = []
dist = 2.5
pitch = -35.0

# Start video recording
# p.setRealTimeSimulation(0) 
ax = 0; fig = 0; fp = []; tp = []
if args.display:
      ax, fp, tp = initDisplay(args.display)
elif args.logging:
      fig = initLogging()
camX, camY = 0, 0

# Mention names of objects
mentionNames(id_lookup)

# Save state
world_states = []
id1 = p.saveState()
world_states.append(id1)
print(id_lookup)
print(fixed_orientation)

# Check Logging
if args.logging:
    deleteAll("logs")
  
if args.display:
    deleteAll("logs\\fp")
    deleteAll("logs\\tp")


# Start simulation
if True:
    while(True):
        camTargetPos = [x1, y1, 0]
        if args.logging or args.display:
          lastTime, imageCount, im = saveImage(lastTime, imageCount, args.logging, args.display, ax, o1, fp, tp, dist, yaw, pitch, camTargetPos)
          if args.logging and im:
                ims.append([im])
        x1, y1, o1, keyboard = moveKeyboard(x1, y1, o1, [husky, robotID])
        moveUR5Keyboard(robotID, wings, gotoWing)
        z1, y1, o1, world_states = restoreOnKeyboard(world_states, x1, y1, o1)
        keepHorizontal(horizontal_list)
        keepOnGround(ground_list)
        keepOrientation(fixed_orientation)
        dist, yaw, pitch, camX, camY = changeCameraOnKeyboard(dist, yaw, pitch, camX, camY)

        p.stepSimulation()  
        # print(checkGoal(goal_file, constraints, states, id_lookup))

        if action_index >= len(actions):
          continue

        if(actions[action_index][0] == "move"):
          target = actions[action_index][1]
          x1, y1, o1, done = move(x1, y1, o1, [husky, robotID], target, keyboard, speed)

        if(actions[action_index][0] == "moveZ"):
          target = actions[action_index][1]
          x1, y1, o1, done = move(x1, yd1, o1, [husky, robotID], target, keyboard, speed, up=True)

        elif(actions[action_index][0] == "moveTo"):
          target = actions[action_index][1]
          x1, y1, o1, done = moveTo(x1, y1, o1, [husky, robotID], id_lookup[target], 
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
            if checkUR5constrained(constraints) and actions[action_index][2] == 'ur5':
                raise Exception("Gripper is not free, can not hold object")
            if (checkInside(constraints, states, id_lookup, actions[action_index][1], enclosures) 
                and actions[action_index][2] == 'ur5'):
                raise Exception("Object is inside an enclosure, can not grasp it.")
            if (actions[action_index][2] in enclosures
                and isClosed(actions[action_index][2], states, id_lookup)):
                raise Exception("Enclosure is closed, can not place object inside")
            cid = constrain(actions[action_index][1], 
                            actions[action_index][2], 
                            cons_link_lookup, 
                            cons_pos_lookup,
                            id_lookup,
                            constraints,
                            ur5_dist)
            constraints[actions[action_index][1]] = (actions[action_index][2], cid)
            waiting = True

        elif(actions[action_index][0] == "removeConstraint"):
          if time.time()-startTime > 1:
            done = True; waiting = False
          if not waiting and not done:
            removeConstraint(constraints, actions[action_index][1], actions[action_index][2])
            del constraints[actions[action_index][1]]
            waiting = True

        elif(actions[action_index][0] == "changeState"):
          if checkUR5constrained(constraints):
              raise Exception("Gripper is not free, can not change state")
          state = actions[action_index][2]
          if state == "stuck" and not actions[action_index][1] in sticky:
              raise Exception("Object not sticky")  
          done = changeState(id_lookup[actions[action_index][1]], states[actions[action_index][1]][state])   

        elif(actions[action_index][0] == "addTo"):
          obj = actions[action_index][1]
          if actions[action_index][2] == "sticky":
            sticky.append(obj) 
          done = True 

        elif(actions[action_index][0] == "saveBulletState"):
          id1 = p.saveState()
          world_states.append(id1)
          done = True

        if done:
          startTime = time.time()
          action_index += 1
          if action_index < len(actions):
            print("Executing action: ", actions[action_index])
          done = False

    p.disconnect()
# except Exception as e: 
#     print(e)
#     p.disconnect()
# finally:
#   if args.logging:
#     ani = animation.ArtistAnimation(fig, ims, interval=500, blit=True,
#                                 repeat_delay=2000)
#     ani.save('logs/action_video.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
                    

