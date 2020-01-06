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
from src.datapoint import Datapoint
import math
import pickle

object_file = "jsons/objects.json"
wings_file = "jsons/wings.json"
tolerance_file = "jsons/tolerance.json"
goal_file = "jsons/goal.json"

#Number of steps before image capture
COUNTER_MOD = 50

# Enclosures
enclosures = ['fridge', 'cupboard']

# Semantic objects
# Sticky objects
sticky = []
# Fixed objects
fixed = []
# Has cleaner
cleaner = False

# Connect to Bullet using GUI mode
light = p.connect(p.GUI)
lightOn = True

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
print ("The world file is", args.world)

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
p.setRealTimeSimulation(1) 
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

# Initialize datapoint
datapoint = Datapoint()
 

def changeView(direction):
  global x1, y1, o1, world_states, dist, yaw, pitch, camX, camY, imageCount, perspective, lightOn
  camTargetPos = [x1, y1, 0]
  dist = dist - 0.5 if direction == "in" else dist + 0.5 if direction == "out" else dist
  yaw = yaw - 25 if direction == "left" else yaw + 25 if direction == "right" else yaw
  print(0, imageCount, perspective, ax, o1, cam, dist, yaw, pitch, camTargetPos)
  perspective = "tp" if perspective == "fp" and direction == None else "fp" if direction == None else perspective
  lastTime, imageCount = saveImage(0, imageCount, perspective, ax, o1, cam, dist, yaw, pitch, camTargetPos, wall_id, lightOn)


def showObject(obj):
  global world_states, x1, y1, o1, imageCount, lightOn
  ((x, y, z), (a1, b1, c1, d1)) = p.getBasePositionAndOrientation(id_lookup[obj])
  _, imageCount = saveImage(0, imageCount, 'fp', ax, math.atan2(y,x)%(2*math.pi), cam, 2, yaw, pitch, [x, y, z], wall_id, lightOn)
  time.sleep(0.5)
  _, imageCount = saveImage(0, imageCount, 'fp', ax, math.atan2(y,x)%(2*math.pi), cam, 7, yaw, pitch, [x, y, z], wall_id, lightOn)
  time.sleep(1)
  firstImage()

def undo():
  global world_states, x1, y1, o1, imageCount, constraints, lightOn, datapoint
  datapoint.addSymbolicAction("Undo")
  datapoint.addPoint(None, None, None, None, 'Undo', None, None, None)
  x1, y1, o1, constraints, world_states = restoreOnInput(world_states, x1, y1, o1, constraints)
  _, imageCount = saveImage(0, imageCount, perspective, ax, o1, cam, dist, yaw, pitch, camTargetPos, wall_id, lightOn)

def firstImage():
  global x1, y1, o1, world_states, dist, yaw, pitch, camX, camY, imageCount, lightOn
  camTargetPos = [x1, y1, 0]
  _, imageCount= saveImage(-250, imageCount, perspective, ax, o1, cam, dist, 50, pitch, camTargetPos, wall_id, lightOn)

def executeHelper(actions, goal_file=None):
  global x1, y1, o1, world_states, dist, yaw, pitch, camX, camY, imageCount, cleaner, lightOn, datapoint, dirtClean
  # List of low level actions
  datapoint.addSymbolicAction(actions['actions'])
  actions = convertActions(actions)
  print(actions)
  action_index = 0
  done = False
  waiting = False
  startTime = time.time()
  lastTime = startTime
  datapoint.addPoint([x1, y1, 0, o1], sticky, fixed, cleaner, 'Start', constraints, getAllPositionsAndOrientations(id_lookup), lightOn, dirtClean)

  # Start simulation
  if True:
      # start_here = time.time()
      counter = 0
      while(True):
          counter += 1
          camTargetPos = [x1, y1, 0]
          if (args.logging or args.display) and (counter % COUNTER_MOD == 0):
            # start_image = time.time()
            lastTime, imageCount = saveImage(lastTime, imageCount, "fp", ax, o1, cam, 3, yaw, pitch, camTargetPos, wall_id, lightOn)
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
          # print(checkGoal(goal_file, constraints, states, id_lookup), constraints)

          if action_index >= len(actions):
            yaw = 180*(math.atan2(y1,x1)%(2*math.pi))/math.pi - 90
            lastTime, imageCount = saveImage(lastTime, imageCount, perspective, ax, o1, cam, dist, yaw, pitch, camTargetPos, wall_id, lightOn)
            return checkGoal(goal_file, constraints, states, id_lookup, lightOn, dirtClean)

          if(actions[action_index][0] == "move"):
            if "husky" in fixed:
              raise Exception("Husky can not move as it is on a stool")    
            target = actions[action_index][1]
            x1, y1, o1, done = move(x1, y1, o1, [husky, robotID], target, keyboard, speed)

          elif(actions[action_index][0] == "moveZ"):
            if "husky" in fixed:
                  raise Exception("Husky can not move as it is on a stool")    
            target = actions[action_index][1]
            x1, y1, o1, done = move(x1, y1, o1, [husky, robotID], target, keyboard, speed, up=True)

          elif(actions[action_index][0] == "moveTo"):
            if objDistance("husky", actions[action_index][1], id_lookup) > 2 and "husky" in fixed:
                  raise Exception("Husky can not move as it is on a stool")  
            if abs(p.getBasePositionAndOrientation(id_lookup[actions[action_index][1]])[0][2] - 
              p.getBasePositionAndOrientation(husky)[0][2]) > 1:
                  raise Exception("Object on different height, please use stool")
            target = actions[action_index][1]
            x1, y1, o1, done = moveTo(x1, y1, o1, [husky, robotID], id_lookup[target], 
                                    tolerances[target], 
                                    keyboard,
                                    speed)

          elif(actions[action_index][0] == "moveToXY"):
            if objDistance("husky", actions[action_index][1], id_lookup) > 2 and "husky" in fixed:
                  raise Exception("Husky can not move as it is on a stool")  
            target = actions[action_index][1]
            x1, y1, o1, done = moveTo(x1, y1, o1, [husky, robotID], id_lookup[target], 
                                    tolerances[target], 
                                    keyboard,
                                    speed)

          elif(actions[action_index][0] == "changeWing"):
            if time.time()-startTime > 1.8:
              done = True
            pose = actions[action_index][1]
            gotoWing(robotID, wings[pose])

          elif(actions[action_index][0] == "constrain"):
            if time.time()-startTime > 1:
              done = True; waiting = False
            if not waiting and not done:
              if checkUR5constrained(constraints) and actions[action_index][2] == 'ur5':
                  raise Exception("Gripper is not free, can not hold object")
              if actions[action_index][2] == actions[action_index][1]:
                  raise Exception("Cant place object on itself")
              if (checkInside(constraints, states, id_lookup, actions[action_index][1], enclosures) 
                  and actions[action_index][2] == 'ur5'):
                  raise Exception("Object is inside an enclosure, can not grasp it.")
              if (actions[action_index][2] in enclosures
                  and isClosed(actions[action_index][2], states, id_lookup)):
                  raise Exception("Enclosure is closed, can not place object inside")
              if (actions[action_index][2] == 'ur5' 
                  and(objDistance(actions[action_index][1], actions[action_index][2], id_lookup)) > 2):
                  raise Exception("Object too far away, move closer to it")
              if ("mop" in actions[action_index][1] 
                  or "sponge" in actions[action_index][1] 
                  or "vacuum" in actions[action_index][1]):
                  cleaner = True
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
              cleaner = False
              removeConstraint(constraints, actions[action_index][1], actions[action_index][2])
              del constraints[actions[action_index][1]]
              waiting = True

          elif(actions[action_index][0] == "changeState"):
            if checkUR5constrained(constraints):
                raise Exception("Gripper is not free, can not change state")
            state = actions[action_index][2]
            if state == "stuck" and not actions[action_index][1] in sticky:
                raise Exception("Object not sticky")  
            if actions[action_index][1] == 'light':
              lightOn = False
              done = True
            else:
              done = changeState(id_lookup[actions[action_index][1]], states[actions[action_index][1]][state])   

          elif(actions[action_index][0] == "climbUp"):
            target = id_lookup[actions[action_index][1]]
            (x2, y2, z2), _ = p.getBasePositionAndOrientation(target)
            targetLoc = [x2, y2, z2+0.4]
            x1, y1, o1, done = move(x1, y1, o1, [husky, robotID], targetLoc, keyboard, speed, tolerance=0.1, up=True)
          
          elif(actions[action_index][0] == "climbDown"):
            target = id_lookup[actions[action_index][1]]
            (x2, y2, z2), _ = p.getBasePositionAndOrientation(target)
            targetLoc = [x2, y2+(2 if y2 < 0 else -2), 0]
            x1, y1, o1, done = move(x1, y1, o1, [husky, robotID], targetLoc, keyboard, speed, up=True)

          elif(actions[action_index][0] == "clean"):
            if not cleaner:
                raise Exception("No cleaning agent with the robot")
            p.changeVisualShape(id_lookup[actions[action_index][1]], -1, rgbaColor = [1, 1, 1, 0])
            dirtClean = True
            done = True

          elif(actions[action_index][0] == "addTo"):
            obj = actions[action_index][1]
            if actions[action_index][2] == "sticky":
              sticky.append(obj) 
            elif actions[action_index][2] == "fixed":
              fixed.append(obj) 
            done = True 
          
          elif(actions[action_index][0] == "removeFrom"):
            obj = actions[action_index][1]
            if actions[action_index][2] == "sticky":
              sticky.remove(obj) 
            elif actions[action_index][2] == "fixed":
              fixed.remove(obj) 
            done = True 

          elif(actions[action_index][0] == "saveBulletState"):
            id1 = p.saveState()
            world_states.append([id1, x1, y1, o1, constraints])
            done = True

          if done:
            startTime = time.time()
            if not actions[action_index][0] == "saveBulletState":
              datapoint.addPoint([x1, y1, 0, o1], sticky, fixed, cleaner, actions[action_index], constraints, getAllPositionsAndOrientations(id_lookup), lightOn, dirtClean)
            action_index += 1
            if action_index < len(actions):
              print("Executing action: ", actions[action_index])
            done = False
          # total_time_taken = time.time() - start_here
          # print ("Total", total_time_taken)
          # print ("Fraction", image_save_time/total_time_taken)
          # start_here = time.time()

def execute(actions, goal_file=None):
  global datapoint
  try:
    return executeHelper(actions, goal_file)
  except Exception as e:
    datapoint.addSymbolicAction("Error = " + str(e))
    datapoint.addPoint(None, None, None, None, 'Error = ' + str(e), None, None, None, None)
    raise e

def saveDatapoint(filename):
  global datapoint
  f = open(filename + '.datapoint', 'wb')
  pickle.dump(datapoint, f)
  f.flush()
  f.close()

def getDatapoint():
  return datapoint

def destroy():
  p.disconnect()
                      

