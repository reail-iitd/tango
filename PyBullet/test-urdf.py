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
from operator import sub
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
# Has stick
stick = False

# Connect to Bullet using GUI mode
light = p.connect(p.GUI)
lightOn = True

# Add input arguments
args = initParser()
speed = args.speed

# if (args.logging or args.display):
#   p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
#   p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
#   p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)

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
 
# Print manipulation region bounding boxes
for obj in id_lookup.keys():
  box = p.getAABB(id_lookup[obj])
  # print(obj, box) # Bounding Box
  print(obj, (abs(box[0][0]-box[1][0]), abs(box[0][1]-box[1][1]), abs(box[0][2]-box[1][2]))) # L, w, h


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
  datapoint.addPoint(None, None, None, None, 'Undo', None, None, None, None, None)
  x1, y1, o1, constraints, world_states = restoreOnInput(world_states, x1, y1, o1, constraints)
  _, imageCount = saveImage(0, imageCount, perspective, ax, o1, cam, dist, yaw, pitch, camTargetPos, wall_id, lightOn)

def firstImage():
  global x1, y1, o1, world_states, dist, yaw, pitch, camX, camY, imageCount, lightOn
  camTargetPos = [x1, y1, 0]
  _, imageCount= saveImage(-250, imageCount, perspective, ax, o1, cam, dist, 50, pitch, camTargetPos, wall_id, lightOn)

def executeHelper(actions, goal_file=None):
  global x1, y1, o1, world_states, dist, yaw, pitch, camX, camY, imageCount, cleaner, lightOn, datapoint, dirtClean, stick
  # List of low level actions
  action_index = 0
  done = False; done1 = False
  waiting = False
  # Start simulation
  if True:
      # start_here = time.time()
      counter = 0
      while(True):
          counter += 1
          camTargetPos = [x1, y1, 0]
          # if (args.logging or args.display) and (counter % COUNTER_MOD == 0):
            # start_image = time.time()
            # lastTime, imageCount = saveImage(lastTime, imageCount, "fp", ax, o1, cam, 3, yaw, pitch, camTargetPos, wall_id, lightOn)
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
                     

# urdf = 'models/urdf/drill.urdf'
# position = [2, 0, 1]
# object_id = p.loadURDF(urdf, list(position))
executeHelper([]) 

