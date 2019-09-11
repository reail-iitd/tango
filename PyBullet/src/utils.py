import pybullet as p
import math
import operator 
import json
from scipy.spatial import distance
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time 
import numpy as np
current_milli_time = lambda: int(round(time.time() * 1000))

camTargetPos = [0, 0, 0]
cameraUp = [0, 0, 2]
cameraPos = [1, 1, 5]
roll = -30
upAxisIndex = 2
camDistance = 5
pixelWidth = 352
pixelHeight = 240
aspect = pixelWidth / pixelHeight
nearPlane = 0.01
farPlane = 100
fov = 60
img_arr = []; img_arr2 = []

def initDisplay(display):
    plt.ion()
    plt.axis('off')
    fp = []; tp = []
    if display == "fp":
        fp = plt.imshow([[1, 2, 3] * 50] * 100, interpolation='none', animated=True)
    if display == "tp":
        tp = plt.imshow([[1, 2, 3] * 50] * 100, interpolation='none', animated=True)
    if  display == "both":
        plt.figure(1)
        fp = plt.imshow([[1, 2, 3] * 50] * 100, interpolation='none', animated=True)
        plt.figure(2)
        tp = plt.imshow([[1, 2, 3] * 50] * 100, interpolation='none', animated=True)
    plt.axis('off')
    ax = plt.gca()
    return ax, fp, tp

def initLogging():
    plt.axis('off')
    fig = plt.figure(figsize = (38.42,21.6))
    return fig

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
    if ord(b'm') in keys:
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

def changeCameraOnKeyboard(camDistance, yaw, pitch, x,y):
    """
    Change camera zoom or angle from keyboard
    """
    mouseEvents = p.getMouseEvents()
    keys = p.getKeyboardEvents()
    if ord(b'a') in keys:
        camDistance += 0.01
    elif ord(b'd') in keys:
        camDistance -= 0.01
    if ord(b'm') not in keys:
        if 65297 in keys:
            pitch += 0.2
        if 65298 in keys:
            pitch -= 0.2
        if 65295 in keys:
            yaw += 0.2
        if 65296 in keys:
            yaw -= 0.2
    return camDistance, yaw, pitch, 0,0


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

def saveImage(lastTime, imageCount, save, display, ax, o1, fp, tp, dist, yaw, pitch, camTargetPos):
    current = current_milli_time()
    if (current - lastTime) < 150:
        return lastTime, imageCount, None
    projectionMatrix = p.computeProjectionMatrixFOV(fov, aspect, nearPlane, farPlane)
    img_arr = []; img_arr2 = []
    if display == "fp" or display == "both":
        viewMatrixFP = p.computeViewMatrixFromYawPitchRoll(camTargetPos, dist, -90+(o1*180/math.pi), -35,
                                                                roll, upAxisIndex)
        img_arr = p.getCameraImage(pixelWidth,
                                      pixelHeight,
                                      viewMatrixFP,
                                      projectionMatrix,
                                      shadow=1,
                                      lightDirection=[1, 1, 1],
                                      renderer=p.ER_BULLET_HARDWARE_OPENGL,
                                      flags=p.ER_NO_SEGMENTATION_MASK)
    if display == "tp" or display == "both":
        viewMatrixTP = p.computeViewMatrixFromYawPitchRoll([0,0,0], 5, yaw, pitch,
                                                            roll, upAxisIndex)
        img_arr2 = p.getCameraImage(pixelWidth,
                                      pixelHeight,
                                      viewMatrixTP,
                                      projectionMatrix,
                                      shadow=1,
                                      lightDirection=[1, 1, 1],
                                      renderer=p.ER_BULLET_HARDWARE_OPENGL,
                                      flags=p.ER_NO_SEGMENTATION_MASK)

    if display:
        if display == "fp" or display == "both":
            rgbFP = img_arr[2]
            fp.set_data(np.reshape(rgbFP, (pixelHeight, pixelWidth, 4)) * (1. / 255.))
        if display == "tp" or display == "both":
            rgbTP = img_arr2[2]
            tp.set_data(np.reshape(rgbTP, (pixelHeight, pixelWidth, 4)) * (1. / 255.))
        ax.plot([0])
        plt.pause(0.00001)
    if save:
        return current, imageCount+1, plt.imshow(rgbTP,interpolation='none')
    return current, imageCount+1, None
