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
import math

object_file = "jsons/objects.json"
wings_file = "jsons/wings.json"
tolerance_file = "jsons/tolerance.json"
goal_file = "jsons/goal.json"

#Number of steps before image capture
COUNTER_MOD = 50

# Enclosures
enclosures = ['fridge', 'cupboard']

# Sticky objects
sticky = []

# Connect to Bullet using GUI mode
light = p.connect(p.GUI)

# Add input arguments
args = initParser()
speed = args.speed

if (args.logging or args.display):
  p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
  p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
  p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)

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

# Init camera
imageCount = 0
yaw = 50
ims = []
dist = 5
pitch = -35.0

# Start video recording
# p.setRealTimeSimulation(0) 
ax = 0; fig = 0; cam = []
if args.display:
      ax, cam = initDisplay("both")
elif args.logging:
      fig = initLogging()
camX, camY = 0, 0

# Mention names of objects
mentionNames(id_lookup)

# Save state
world_states = []
id1 = p.saveState()
world_states.append([id1, x1, y1, o1, constraints])
print(id_lookup)
print(fixed_orientation)

# Check Logging
if args.logging or args.display:
    deleteAll("logs")

# Default perspective
perspective = "tp"

# Wall to make trasparent when camera outside
wall_id = -1
if 'home' in args.world:
  wall_id = id_lookup['walls']
 
def changeView(direction):
  global x1, y1, o1, world_states, dist, yaw, pitch, camX, camY, imageCount, perspective
  camTargetPos = [x1, y1, 0]
  dist = dist - 0.5 if direction == "in" else dist + 0.5 if direction == "out" else dist
  yaw = yaw - 5 if direction == "left" else yaw + 5 if direction == "right" else yaw
  print(0, imageCount, perspective, ax, o1, cam, dist, yaw, pitch, camTargetPos)
  perspective = "tp" if perspective == "fp" and direction == None else "fp" if direction == None else perspective
  lastTime, imageCount = saveImage(0, imageCount, perspective, ax, o1, cam, dist, yaw, pitch, camTargetPos, wall_id)


def showObject(obj):
  global world_states, x1, y1, o1, imageCount
  ((x, y, z), (a1, b1, c1, d1)) = p.getBasePositionAndOrientation(id_lookup[obj])
  saveImage(0, imageCount, 'fp', ax, math.atan2(y,x)%(2*math.pi), cam, 2, yaw, pitch, [x, y, z], wall_id)

def undo():
  global world_states, x1, y1, o1, imageCount, constraints
  x1, y1, o1, constraints, world_states = restoreOnInput(world_states, x1, y1, o1, constraints)
  saveImage(0, imageCount, perspective, ax, o1, cam, dist, yaw, pitch, camTargetPos, wall_id)

def firstImage():
  global x1, y1, o1, world_states, dist, yaw, pitch, camX, camY, imageCount
  camTargetPos = [x1, y1, 0]
  lastTime, imageCount= saveImage(-250, imageCount, perspective, ax, o1, cam, dist, 50, pitch, camTargetPos, wall_id)

def execute(actions):
  global x1, y1, o1, world_states, dist, yaw, pitch, camX, camY, imageCount
  # List of low level actions
  actions = convertActions(actions)
  print(actions)
  action_index = 0
  done = False
  waiting = False
  startTime = current_milli_time()
  lastTime = startTime

  # Start simulation
  if True:
      # start_here = time.time()
      counter = 0
      while(True):
          counter += 1
          camTargetPos = [x1, y1, 0]
          if (args.logging or args.display) and (counter % COUNTER_MOD == 0):
            # start_image = time.time()
            lastTime, imageCount = saveImage(lastTime, imageCount, "fp", ax, o1, cam, dist, yaw, pitch, camTargetPos, wall_id)
            # image_save_time = time.time() - start_image
            # print ("Image save time", image_save_time)
          x1, y1, o1, keyboard = moveKeyboard(x1, y1, o1, [husky, robotID])
          moveUR5Keyboard(robotID, wings, gotoWing)
          # x1, y1, o1, world_states = restoreOnKeyboard(world_states, x1, y1, o1)
          keepHorizontal(horizontal_list)
          keepOnGround(ground_list)
          keepOrientation(fixed_orientation)
          # dist, yaw, pitch, camX, camY = changeCameraOnKeyboard(dist, yaw, pitch, camX, camY)

          # start = time.time()
          p.stepSimulation() 
          # print ("Step simulation time ",time.time() - start) 
          # print(checkGoal(goal_file, constraints, states, id_lookup))

          if action_index >= len(actions):
            lastTime, imageCount = saveImage(lastTime, imageCount, "fp", ax, o1, cam, dist, yaw, pitch, camTargetPos, wall_id)
            break

          if(actions[action_index][0] == "move"):
            target = actions[action_index][1]
            x1, y1, o1, done = move(x1, y1, o1, [husky, robotID], target, keyboard, speed)

          elif(actions[action_index][0] == "moveZ"):
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
            world_states.append([id1, x1, y1, o1, constraints])
            done = True

          if done:
            startTime = time.time()
            action_index += 1
            if action_index < len(actions):
              print("Executing action: ", actions[action_index])
            done = False
          # total_time_taken = time.time() - start_here
          # print ("Total", total_time_taken)
          # print ("Fraction", image_save_time/total_time_taken)
          # start_here = time.time()

def destroy():
  p.disconnect()
                      

