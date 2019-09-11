import pybullet as p
import math
import operator 
import json
from scipy.spatial import distance
import matplotlib.pyplot as plt
import time 
import numpy as np
current_milli_time = lambda: int(round(time.time() * 1000))

plt.ion()
plt.axis('off')
ax = plt.gca()

camTargetPos = [0, 0, 2]
cameraUp = [0, 0, 2]
cameraPos = [1, 1, 20]
pitch = -20.0
roll = -30
upAxisIndex = 2
camDistance = 4
pixelWidth = 1920
pixelHeight = 1080
nearPlane = 0.01
farPlane = 100
fov = 60
yaw = 40
img = [[1, 2, 3] * 50] * 100  #np.random.rand(200, 320)
image = plt.imshow(img, interpolation='none', animated=True, label="blah")
names = {}

def keepHorizontal(object_list):
    """
    Keep the objects horizontal
    """
    for obj_id in object_list:
        p.resetBasePositionAndOrientation(obj_id,
                                          p.getBasePositionAndOrientation(obj_id)[0],
                                          p.getQuaternionFromEuler((0,0,0)))

def keepOnGround(object_list):
    """
    Keep the objects on ground
    """
    for obj_id in object_list:
        p.resetBasePositionAndOrientation(obj_id,
                                          (p.getBasePositionAndOrientation(obj_id)[0][0],
                                          p.getBasePositionAndOrientation(obj_id)[0][1], 0.01),
                                          p.getBasePositionAndOrientation(obj_id)[1])

def keepOrientation(objects):
    """
    keeps the orientation fixed
    """
    for obj_id in objects.keys():
        p.resetBasePositionAndOrientation(obj_id,
                                          p.getBasePositionAndOrientation(obj_id)[0],
                                          objects[obj_id])

def moveKeyboard(x1, y1, o1, object_list):
    """
    Move robot based on keyboard inputs
    """
    flag = False; delz = 0
    keys = p.getKeyboardEvents()
    if 65297 in keys:
        x1 += math.cos(o1)*0.001
        y1 += math.sin(o1)*0.001
        flag= True
    if 65298 in keys:
        x1 -= math.cos(o1)*0.001
        y1 -= math.sin(o1)*0.001
        flag= True
    if ord(b'o') in keys:
        delz = 0.001
        flag = True
    if ord(b'l') in keys:
        delz = -0.001
        flag = True
    if 65295 in keys:
        o1 += 0.005
        flag= True
    if 65296 in keys:
        o1 -= 0.005
        flag= True
    q = p.getQuaternionFromEuler((0,0,o1))
    for obj_id in object_list:
        (x, y, z1) = p.getBasePositionAndOrientation(obj_id)[0]
        z1 = max(0, z1+delz)
        if p.getBasePositionAndOrientation(obj_id)[0] != ((x1, y1, z1), (q)):
            p.resetBasePositionAndOrientation(obj_id, [x1, y1, z1], q)
    return  x1, y1, o1, flag

def moveUR5Keyboard(robotID, wings, gotoWing):
    """
    Change UR5 arm position based on keyboard input
    """
    keys = p.getKeyboardEvents()
    if ord(b'h') in keys:
        gotoWing(robotID, wings["home"])
        return
    if ord(b'u') in keys:
        gotoWing(robotID, wings["up"])
        return
    if ord(b'n') in keys:
        gotoWing(robotID, wings["down"])
    return

def changeYaw(yaw):
    """
    Change yaw of camera
    """
    keys = p.getKeyboardEvents()
    if ord(b'a') in keys:
        return yaw - 1
    if ord(b'd') in keys:
        return yaw + 1
    return yaw

def mentionNames(id_lookup):
    """
    Add labels of all objects in the world
    """
    if len(names.keys()) == 0:
        for obj in id_lookup.keys():
            id = p.addUserDebugText(obj, 
                            (0, 0, 0.2),
                            parentObjectUniqueId=id_lookup[obj])


def restoreOnKeyboard(world_states, x1, y1, o1):
    """
    Restore to last saved state when 'r' is pressed
    """
    keys = p.getKeyboardEvents()
    if ord(b'r') in keys:
        if len(world_states) != 0:
            id1 = world_states.pop()
            p.restoreState(stateId=id1)
            q=p.getQuaternionFromEuler((0,0,0))
            p.resetBasePositionAndOrientation(([0, 0, 0], q)) # Get robot to home when undo
        return 0, 0, 0, world_states
    return x1, y1, o1, world_states


def checkGoal(goal_file, constraints, states, id_lookup):
    """
    Check if goal conditions are true for the current state
    """
    with open(goal_file, 'r') as handle:
        file = json.load(handle)
    goals = file['goals']
    success = True

    for goal in goals:
        obj = goal['object']
        if goal['target'] != "":
            constrained = False
            for obj1 in constraints.keys():
                if obj1 == obj and constraints[obj][0] == goal["target"]:
                    constrained = True
            success = success and constrained

        if goal['state'] != "":
            positionAndOrientation = states[obj][goal['state']]
            q=p.getQuaternionFromEuler(positionAndOrientation[1])
            ((x1, y1, z1), (a1, b1, c1, d1)) = p.getBasePositionAndOrientation(id_lookup[obj])
            ((x2, y2, z2), (a2, b2, c2, d2)) = (positionAndOrientation[0], q)
            done = (True and 
                abs(x2-x1) <= 0.01 and 
                abs(y2-y1) <= 0.01 and 
                abs(a2-a1) <= 0.01 and 
                abs(b2-b2) <= 0.01 and 
                abs(c2-c1) <= 0.01 and 
                abs(d2-d2) <= 0.01)
            success = success and done

        if goal['position'] != []:
            pos = p.getBasePositionAndOrientation(id_lookup[obj])[0]
            goal_pos = goal['position']
            if abs(distance.euclidean(pos, goal_pos)) > abs(goal['tolerance']):
                success = False
    return success

def checkUR5constrained(constraints):
    """
    Check if UR5 gripper is already holding something
    """
    for obj in constraints.keys():
        if constraints[obj][0] == 'ur5':
            return True
    return False

def checkInside(constraints, states, id_lookup, obj, enclosures):
    """
    Check if object is inside cupboard or fridge
    """
    if obj in constraints.keys():
        for enclosure in enclosures:
            if constraints[obj][0] == enclosure:
                positionAndOrientation = states[enclosure]["close"]
                q=p.getQuaternionFromEuler(positionAndOrientation[1])
                ((x1, y1, z1), (a1, b1, c1, d1)) = p.getBasePositionAndOrientation(id_lookup[enclosure])
                ((x2, y2, z2), (a2, b2, c2, d2)) = (positionAndOrientation[0], q)
                closed = (abs(x2-x1) <= 0.01 and 
                        abs(y2-y1) <= 0.01 and 
                        abs(a2-a1) <= 0.01 and 
                        abs(b2-b2) <= 0.01 and 
                        abs(c2-c1) <= 0.01 and 
                        abs(d2-d2) <= 0.01)
                if closed:
                    return True
    return False

def isClosed(enclosure, states, id_lookup):
    """
    Check if enclosure is closed or not
    """
    positionAndOrientation = states[enclosure]["close"]
    q=p.getQuaternionFromEuler(positionAndOrientation[1])
    ((x1, y1, z1), (a1, b1, c1, d1)) = p.getBasePositionAndOrientation(id_lookup[enclosure])
    ((x2, y2, z2), (a2, b2, c2, d2)) = (positionAndOrientation[0], q)
    closed = (abs(x2-x1) <= 0.01 and 
            abs(y2-y1) <= 0.01 and 
            abs(a2-a1) <= 0.01 and 
            abs(b2-b2) <= 0.01 and 
            abs(c2-c1) <= 0.01 and 
            abs(d2-d2) <= 0.01)
    return closed

def saveImage(lastTime, imageCount, yaw):
    current = current_milli_time()
    if (current - lastTime) < 500:
        return lastTime, imageCount
    viewMatrix = p.computeViewMatrixFromYawPitchRoll(camTargetPos, camDistance, yaw, pitch,
                                                            roll, upAxisIndex)
    aspect = pixelWidth / pixelHeight
    projectionMatrix = p.computeProjectionMatrixFOV(fov, aspect, nearPlane, farPlane)
    img_arr = p.getCameraImage(pixelWidth,
                                      pixelHeight,
                                      viewMatrix,
                                      projectionMatrix,
                                      shadow=1,
                                      lightDirection=[1, 1, 1],
                                      renderer=p.ER_BULLET_HARDWARE_OPENGL)
    w = img_arr[0]  #width of the image, in pixels
    h = img_arr[1]  #height of the image, in pixels
    rgb = img_arr[2]  #color data RGB
    dep = img_arr[3]  #depth data
    np_img_arr = np.reshape(rgb, (h, w, 4))
    np_img_arr = np_img_arr * (1. / 255.)

    image.set_data(np_img_arr)
    ax.plot([0])
    plt.savefig("logs/" + str(imageCount) + ".png")
    return current, imageCount+1
